package com.irhammuch.android.facerecognition;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import org.tensorflow.lite.support.common.FileUtil;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private Interpreter tflite;
    private ImageView resultView;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultView = findViewById(R.id.imageView);

        try {
            loadModel();
            runInference();
        } catch (Exception e) {
            Log.e(TAG, "Error in onCreate", e);
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void loadModel() throws IOException {
        try {
            MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(this, "yolov8_3.tflite");
            Interpreter.Options options = new Interpreter.Options();
            tflite = new Interpreter(modelBuffer, options);
            Log.d(TAG, "Model loaded successfully");
        } catch (IOException e) {
            Log.e(TAG, "Error loading model", e);
            Toast.makeText(this, "Failed to load model: " + e.getMessage(), Toast.LENGTH_LONG).show();
            throw e;
        }
    }

    private void runInference() {
        try {
            Log.d(TAG, "Starting inference...");

            Bitmap frame = getDummyFrame();
            Bitmap resized = Bitmap.createScaledBitmap(frame, 640, 640, true);
            Log.d(TAG, "Resized bitmap to 640x640");

            int batchSize = 4;
            int imageSize = 640;
            int pixelSize = 3;
            int inputSize = batchSize * imageSize * imageSize * pixelSize * 4;  // float32 = 4 bytes
            Log.d(TAG, "Allocating input buffer of size: " + inputSize);

            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputSize);
            inputBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            resized.getPixels(intValues, 0, imageSize, 0, 0, imageSize, imageSize);

            Log.d(TAG, "Converting pixels to ByteBuffer...");

            for (int i = 0; i < intValues.length; i++) {
                int val = intValues[i];

                inputBuffer.putFloat(((val >> 16) & 0xFF) / 255.f); // R
                inputBuffer.putFloat(((val >> 8) & 0xFF) / 255.f);  // G
                inputBuffer.putFloat((val & 0xFF) / 255.f);         // B
            }

            // Get input shape
            int[] inputShape = tflite.getInputTensor(0).shape();
            Log.d(TAG, "Model input shape: " + java.util.Arrays.toString(inputShape));

            int[] outputShape = tflite.getOutputTensor(0).shape();
            Log.d(TAG, "Model output shape: " + java.util.Arrays.toString(outputShape));

            // Ensure the output shape is [4, 5, 8400] (or adjust according to your model)
            if (outputShape.length == 3 && outputShape[0] == 4 && outputShape[1] == 5 && outputShape[2] == 8400) {
                TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputShape, org.tensorflow.lite.DataType.FLOAT32);

                Log.d(TAG, "Running inference...");

                tflite.run(inputBuffer, outputBuffer.getBuffer().rewind());
                Log.d(TAG, "Inference completed.");

                float[] outputArray = outputBuffer.getFloatArray();
                Log.d(TAG, "Output size: " + outputArray.length);

                // Check if output array matches expected size [4, 5, 8400]
                if (outputArray.length == 168000) {
                    Log.d(TAG, "Output array is correctly sized: " + outputArray.length);

                    // Reshape the output array into 3D array [4, 5, 8400]
                    float[][][] reshapedOutput = new float[4][5][8400];
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 5; j++) {
                            for (int k = 0; k < 8400; k++) {
                                int index = (i * 5 * 8400) + (j * 8400) + k;
                                reshapedOutput[i][j][k] = outputArray[index];
                            }
                        }
                    }

                    // Now you can use reshapedOutput with the correct shape
                    drawResult(resized, reshapedOutput);
                } else {
                    Log.w(TAG, "Output array has an unexpected size: " + outputArray.length);
                }

            } else {
                Log.e(TAG, "Model output shape is not [4, 5, 8400]. Actual shape: " + java.util.Arrays.toString(outputShape));
            }

        } catch (Exception e) {
            Log.e(TAG, "Error running inference", e);
            Toast.makeText(this, "Inference error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void drawResult(Bitmap frame, float[][][] outputArray) {
        try {
            Bitmap mutable = frame.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutable);
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5f);
            paint.setTextSize(40);

            // Duyệt qua các batch (4 mẫu)
            for (int i = 0; i < 4; i++) {
                    // Lấy tọa độ, chiều rộng, chiều cao và confidence score
                    float x = outputArray[i][0][0] * frame.getWidth(); // Tọa độ x
                    float y = outputArray[i][1][0] * frame.getHeight(); // Tọa độ y
                    float width = outputArray[i][2][0] * frame.getWidth(); // Chiều rộng
                    float height = outputArray[i][3][0] * frame.getHeight(); // Chiều cao
                    float confidenceScore = outputArray[i][4][0]; // Độ tin cậy

                    // Vẽ bounding box nếu confidence score > 0.5 (hoặc giá trị tùy chỉnh)
                    if (confidenceScore > 0.5) {
                        canvas.drawRect(x, y, x + width, y + height, paint);

                        // Vẽ tên đối tượng (ở đây tôi giả định tên đối tượng là "Ball")
                        Paint textPaint = new Paint();
                        textPaint.setColor(Color.RED);
                        textPaint.setTextSize(40);
                        textPaint.setStyle(Paint.Style.FILL);
                        canvas.drawText("Ball", x + 25, y + 25, textPaint);
                    }

            }

            resultView.setImageBitmap(mutable);
        } catch (Exception e) {
            Log.e(TAG, "Error drawing result", e);
        }
    }





    private Bitmap getDummyFrame() {
        try {
            return android.graphics.BitmapFactory.decodeResource(getResources(), R.drawable.ball);
        } catch (Exception e) {
            Bitmap bitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(bitmap);
            canvas.drawColor(Color.WHITE);

            Paint paint = new Paint();
            paint.setColor(Color.BLACK);
            paint.setStyle(Paint.Style.FILL);
            canvas.drawCircle(320, 240, 50, paint);

            return bitmap;
        }
    }

    @Override
    protected void onDestroy() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        super.onDestroy();
    }
}