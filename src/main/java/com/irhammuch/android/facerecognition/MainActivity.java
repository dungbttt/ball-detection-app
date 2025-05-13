package com.irhammuch.android.facerecognition;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.widget.ImageView;
import android.widget.MediaController;
import android.widget.Toast;
import android.widget.VideoView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    // TFLite interpreter
    private Interpreter tflite;

    // UI
    private ImageView resultView;
    private VideoView videoView;
    private Handler handler = new Handler();

    // Video frame processing
    private MediaMetadataRetriever retriever;
    private int currentFramePosition = 0;
    private final int frameInterval = 100; // ms
    private boolean isProcessing = false;
    private boolean boxDetected = false;

    // Assets
    private static final String WEIGHT_PATH    = "best_stg1_float32.tflite";
    private static final String VIDEO_FILENAME = "ball_video.mp4";

    // Limits
    private static final int MAX_FRAMES    = 200;
    private static final float CONF_THRESH = 0.5f;
    private static final float IOU_THRESH  = 0.45f;
    private static final int DET_SIZE      = 640;  // model input size

    // Pre-post compute
    private float ratio;
    private int padW, padH;

    // For flat output format
    private static final int BOX_ELEMENTS = 6; // Each box has x, y, w, h, confidence, class_id

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultView = findViewById(R.id.imageView);
        videoView  = findViewById(R.id.videoView);

        try {
            loadModel();         // load TFLite (with XNNPACK disabled)
            setupVideoPlayer();  // prepare video + start processing
        } catch (Exception e) {
            Log.e(TAG, "Error in onCreate", e);
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    // Load model from assets
    private void loadModel() throws IOException {
        MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(this, WEIGHT_PATH);
        Interpreter.Options options = new Interpreter.Options();
        // disable XNNPACK to support INT64 tensors
        options.setUseXNNPACK(false);
        tflite = new Interpreter(modelBuffer, options);
        Log.d(TAG, "Model loaded successfully (XNNPACK disabled)");

        // Debug - Print model input and output shapes
        int[] inputShape = tflite.getInputTensor(0).shape();
        int[] outputShape = tflite.getOutputTensor(0).shape();
        Log.d(TAG, "Model input shape: " + Arrays.toString(inputShape));
        Log.d(TAG, "Model output shape: " + Arrays.toString(outputShape));
    }

    // Set up VideoView and start frame-by-frame processing
    private void setupVideoPlayer() {
        String videoPath = copyVideoFromAssets();
        if (videoPath == null) {
            Toast.makeText(this, "Cannot load video", Toast.LENGTH_LONG).show();
            return;
        }

        MediaController mc = new MediaController(this);
        mc.setAnchorView(videoView);
        videoView.setMediaController(mc);
        videoView.setVideoURI(Uri.parse(videoPath));
        videoView.setOnPreparedListener(mp -> {
            Log.d(TAG, "Video prepared");
            setupVideoRetriever(videoPath);
            startVideoProcessing();
        });
        videoView.setOnCompletionListener(mp -> Log.d(TAG, "Video playback completed"));
    }

    // Copy video.mp4 from assets to external files dir
    private String copyVideoFromAssets() {
        File out = new File(getExternalFilesDir(null), VIDEO_FILENAME);
        if (out.exists()) return out.getAbsolutePath();
        try (InputStream in = getAssets().open(VIDEO_FILENAME);
             OutputStream os = new FileOutputStream(out)) {
            byte[] buf = new byte[1024];
            int r;
            while ((r = in.read(buf)) != -1) os.write(buf, 0, r);
            return out.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, "copyVideoFromAssets failed", e);
            return null;
        }
    }

    // Init retriever to pull frames
    private void setupVideoRetriever(String path) {
        retriever = new MediaMetadataRetriever();
        retriever.setDataSource(path);
        Log.d(TAG, "Retriever ready");
    }

    private void startVideoProcessing() {
        if (isProcessing || retriever == null) return;
        isProcessing = true;
        currentFramePosition = 0;
        processNextFrame();
    }

    private void processNextFrame() {
        if (!isProcessing || currentFramePosition >= MAX_FRAMES) {
            isProcessing = false;
            return;
        }
        Bitmap frame = retriever.getFrameAtTime(
                currentFramePosition * frameInterval * 1000L,
                MediaMetadataRetriever.OPTION_CLOSEST
        );
        if (frame != null) {
            runInference(frame);
            currentFramePosition++;
            handler.postDelayed(this::processNextFrame, frameInterval);
        } else {
            Log.d(TAG, "No more frames");
            isProcessing = false;
        }
    }

    // Compute resize ratio & pad for original dims
    private void calculateResizeParams(int origW, int origH) {
        ratio = Math.min((float) DET_SIZE / origW, (float) DET_SIZE / origH);
        int rw = Math.round(origW * ratio);
        int rh = Math.round(origH * ratio);
        padW = (DET_SIZE - rw) / 2;
        padH = (DET_SIZE - rh) / 2;
    }

    // Resize + pad to square DET_SIZE×DET_SIZE
    private Bitmap resizeAndPad(Bitmap bmp) {
        int ow = bmp.getWidth(), oh = bmp.getHeight();
        calculateResizeParams(ow, oh);
        int rw = Math.round(ow * ratio), rh = Math.round(oh * ratio);
        Bitmap resized = Bitmap.createScaledBitmap(bmp, rw, rh, true);
        Bitmap square  = Bitmap.createBitmap(DET_SIZE, DET_SIZE, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(square);
        c.drawColor(Color.BLACK);
        c.drawBitmap(resized, padW, padH, null);
        return square;
    }

    // Intersection-over-Union of two boxes [x1,y1,x2,y2]
    private float iou(float[] a, float[] b) {
        float x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
        float x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
        float w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
        float inter = w * h;
        float areaA = (a[2]-a[0])*(a[3]-a[1]);
        float areaB = (b[2]-b[0])*(b[3]-b[1]);
        return inter / (areaA + areaB - inter + 1e-6f);
    }

    // Non-Maximum Suppression (CPU)
    private int[] nms(List<float[]> boxes, List<Float> scores) {
        Integer[] idxs = new Integer[boxes.size()];
        for (int i = 0; i < idxs.length; i++) idxs[i] = i;
        Arrays.sort(idxs, (i,j) -> -Float.compare(scores.get(i), scores.get(j)));
        List<Integer> keep = new ArrayList<>();
        boolean[] rem = new boolean[boxes.size()];
        for (int i : idxs) {
            if (rem[i]) continue;
            keep.add(i);
            for (int j : idxs) {
                if (i == j || rem[j]) continue;
                if (iou(boxes.get(i), boxes.get(j)) > IOU_THRESH) rem[j] = true;
            }
        }
        return keep.stream().mapToInt(x -> x).toArray();
    }

    // Scale coords from DET_SIZE back to original image
    private void scaleCoords(float[] box, int ow, int oh) {
        box[0] = (box[0] - padW) / ratio;
        box[1] = (box[1] - padH) / ratio;
        box[2] = (box[2] - padW) / ratio;
        box[3] = (box[3] - padH) / ratio;
        box[0] = Math.max(0, Math.min(box[0], ow));
        box[1] = Math.max(0, Math.min(box[1], oh));
        box[2] = Math.max(0, Math.min(box[2], ow));
        box[3] = Math.max(0, Math.min(box[3], oh));
    }

    private void runInference(Bitmap frame) {
        try {
            // Preprocess
            Bitmap inBmp = resizeAndPad(frame);
            ByteBuffer inBuf = ByteBuffer.allocateDirect(DET_SIZE * DET_SIZE * 3 * 4);
            inBuf.order(ByteOrder.nativeOrder());
            int[] pix = new int[DET_SIZE * DET_SIZE];
            inBmp.getPixels(pix, 0, DET_SIZE, 0, 0, DET_SIZE, DET_SIZE);
            for (int v : pix) {
                inBuf.putFloat(((v >> 16) & 0xFF) / 255f);
                inBuf.putFloat(((v >> 8)  & 0xFF) / 255f);
                inBuf.putFloat((v        & 0xFF) / 255f);
            }
            inBuf.rewind();

            // Get model's expected output shape
            int[] outShape = tflite.getOutputTensor(0).shape();
            Log.d(TAG, "Model output shape: " + Arrays.toString(outShape));

            if (outShape.length < 1) {
                Log.e(TAG, "Invalid output shape: " + Arrays.toString(outShape));
                return;
            }

            // Allocate output buffer with exact byte size needed
            int byteSize = tflite.getOutputTensor(0).numBytes();
            Log.d(TAG, "Output byte size: " + byteSize);

            if (byteSize <= 0) {
                Log.e(TAG, "Invalid output byte size: " + byteSize);
                return;
            }

            // Create output buffer and run inference
            ByteBuffer outBuf = ByteBuffer.allocateDirect(byteSize)
                    .order(ByteOrder.nativeOrder());
            tflite.run(inBuf, outBuf.rewind());

            // Convert output to float array safely
            outBuf.rewind();
            int floatCount = byteSize / 4; // Each float is 4 bytes
            float[] outArr = new float[floatCount];

            // Get all floats from buffer
            FloatBuffer floatBuffer = outBuf.asFloatBuffer();
            if (floatBuffer.remaining() < floatCount) {
                floatCount = floatBuffer.remaining();
                outArr = new float[floatCount];
                Log.w(TAG, "Adjusted float count to buffer remaining: " + floatCount);
            }
            floatBuffer.get(outArr, 0, floatCount);

            // Debug output shape & size
            Log.d(TAG, "Output shape: " + Arrays.toString(outShape) +
                    ", Float count: " + outArr.length);

            // Log the first few values for debugging
            StringBuilder valueLog = new StringBuilder("First values: ");
            for (int i = 0; i < Math.min(10, outArr.length); i++) {
                valueLog.append(outArr[i]).append(", ");
            }
            Log.d(TAG, valueLog.toString());

            // Postprocess: extract boxes & scores
            List<float[]> boxList = new ArrayList<>();
            List<Float> scoreList = new ArrayList<>();

            // Handle flat output format [1, 300]
            handleFlatOutput(outArr, boxList, scoreList, frame.getWidth(), frame.getHeight());

            // NMS + scale back
            if (!boxList.isEmpty()) {
                int[] keep = nms(boxList, scoreList);
                List<float[]> finalBoxes = new ArrayList<>();
                for (int idx : keep) {
                    float[] bb = boxList.get(idx);
                    scaleCoords(bb, frame.getWidth(), frame.getHeight());
                    finalBoxes.add(bb);
                }

                // Draw results
                Bitmap outBmp = frame.copy(Bitmap.Config.ARGB_8888, true);
                Canvas canvas = new Canvas(outBmp);
                Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(5f);
                for (float[] bb : finalBoxes) {
                    canvas.drawRect(bb[0], bb[1], bb[2], bb[3], paint);
                }
                runOnUiThread(() -> resultView.setImageBitmap(outBmp));

                // Play video on first detection
                if (!finalBoxes.isEmpty() && !boxDetected) {
                    boxDetected = true;
                    runOnUiThread(() -> {
                        videoView.start();
                        Toast.makeText(this, "Detected, starting video!", Toast.LENGTH_SHORT).show();
                    });
                }
            } else {
                // No detections - just display original frame
                runOnUiThread(() -> resultView.setImageBitmap(frame));
            }

        } catch (Exception e) {
            Log.e(TAG, "Inference error", e);
            e.printStackTrace();

            // Show error frame but continue processing
            runOnUiThread(() -> {
                resultView.setImageBitmap(frame);
                Toast.makeText(this, "Inference error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            });
        }
    }

    // Handle flat array output format [1, 300]
    private void handleFlatOutput(float[] outArr, List<float[]> boxList, List<Float> scoreList, int origWidth, int origHeight) {
        // First, check if array has enough elements to process
        if (outArr.length <= 1) {
            Log.e(TAG, "Output array is too small: " + outArr.length);
            return;
        }

        try {
            // Log the first few values to understand the format
            StringBuilder sb = new StringBuilder("First values: ");
            for (int i = 0; i < Math.min(10, outArr.length); i++) {
                sb.append(outArr[i]).append(", ");
            }
            Log.d(TAG, sb.toString());

            // Get the actual detection count - might be the first element
            // or we might need to detect valid boxes another way
            int numDetections = 0;

            // If first value is reasonable as a count (between 0 and 50)
            if (outArr[0] >= 0 && outArr[0] <= 50) {
                numDetections = (int) outArr[0];
                Log.d(TAG, "Using first value as detection count: " + numDetections);
            } else {
                // Alternative: scan for valid detections based on confidence values
                // For YOLOv8, each detection typically has format:
                // [x, y, w, h, confidence, class_id]
                for (int i = 0; i + 5 < outArr.length; i += BOX_ELEMENTS) {
                    float conf = outArr[i + 4];
                    if (conf > 0 && conf <= 1.0) {
                        numDetections++;
                    }
                }
                Log.d(TAG, "Found " + numDetections + " potential detections by scanning");
            }

            // Safety check
            int maxPossibleDetections = (outArr.length - 1) / BOX_ELEMENTS;
            if (numDetections > maxPossibleDetections) {
                Log.w(TAG, "Detection count too high, limiting from " + numDetections +
                        " to " + maxPossibleDetections);
                numDetections = maxPossibleDetections;
            }

            // Process the detections
            for (int i = 0; i < numDetections; i++) {
                // Calculate offset based on whether we're using the first value as count
                int offset;
                if (outArr[0] >= 0 && outArr[0] <= 50) {
                    offset = i * BOX_ELEMENTS + 1; // +1 to skip the count
                } else {
                    offset = i * BOX_ELEMENTS;
                }

                // Safety check
                if (offset + 5 >= outArr.length) {
                    Log.w(TAG, "Breaking at detection " + i + ", offset " + offset +
                            " would exceed array length " + outArr.length);
                    break;
                }

                float conf = outArr[offset + 4]; // confidence value
                if (conf < CONF_THRESH) continue;

                // Log detection info
                Log.d(TAG, String.format("Detection %d (offset %d): [%.2f, %.2f, %.2f, %.2f, %.2f]",
                        i, offset,
                        outArr[offset], outArr[offset+1],
                        outArr[offset+2], outArr[offset+3],
                        outArr[offset+4]));

                // Get coordinates
                float x = outArr[offset];
                float y = outArr[offset + 1];
                float w = outArr[offset + 2];
                float h = outArr[offset + 3];

                // Check if values are valid
                if (Float.isNaN(x) || Float.isNaN(y) || Float.isNaN(w) || Float.isNaN(h)) {
                    Log.w(TAG, "Skipping detection with NaN values");
                    continue;
                }

                // Convert to absolute coordinates based on the input size DET_SIZE
                float x1, y1, x2, y2;

                // Check if coordinates are likely normalized (between 0-1)
                boolean isNormalized = (Math.abs(x) <= 1 && Math.abs(y) <= 1 &&
                        Math.abs(w) <= 1 && Math.abs(h) <= 1);

                if (isNormalized) {
                    // Convert from normalized [0-1] to absolute coords
                    x1 = (x - w/2) * DET_SIZE;
                    y1 = (y - h/2) * DET_SIZE;
                    x2 = (x + w/2) * DET_SIZE;
                    y2 = (y + h/2) * DET_SIZE;
                } else {
                    // Assume coords are already in pixels
                    x1 = x - w/2;
                    y1 = y - h/2;
                    x2 = x + w/2;
                    y2 = y + h/2;
                }

                // Skip invalid boxes
                if (x1 >= x2 || y1 >= y2 ||
                        x2 <= 0 || y2 <= 0 ||
                        x1 >= DET_SIZE || y1 >= DET_SIZE) {
                    Log.w(TAG, "Skipping invalid box: " + x1 + "," + y1 + "," + x2 + "," + y2);
                    continue;
                }

                boxList.add(new float[]{x1, y1, x2, y2});
                scoreList.add(conf);

                Log.d(TAG, String.format("Added box %d: [%.1f, %.1f, %.1f, %.1f], conf: %.2f",
                        i, x1, y1, x2, y2, conf));
            }
        } catch (Exception e) {
            Log.e(TAG, "Error processing flat output", e);
            e.printStackTrace();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        isProcessing = false;
        handler.removeCallbacksAndMessages(null);
        if (videoView.isPlaying()) videoView.pause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!isProcessing && retriever != null) startVideoProcessing();
        if (boxDetected && !videoView.isPlaying()) videoView.start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (tflite != null) {
            tflite.close();
            tflite = null;
        }

        if (retriever != null) {
            try {
                retriever.release();
            } catch (IOException e) {
                Log.e(TAG, "Error releasing retriever", e);
            }
            retriever = null;
        }

        handler.removeCallbacksAndMessages(null);
    }
}