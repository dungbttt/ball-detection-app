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

    // Các biến output tương ứng với kết quả suy luận của YOLOv2
    private float[][][] outputLocations = new float[1][5][8400]; // Bounding box locations [batch, num_classes, num_detections]
    private float[][][] outputClasses = new float[1][5][8400];    // Class labels
    private float[][][] outputScores = new float[1][5][8400];     // Confidence scores
    private float[][] numDetections = new float[1][1];             // Number of detections

    private static final String WEIGHT_PATH = "yolov8_3.tflite"; // Đường dẫn tới mô hình

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultView = findViewById(R.id.imageView);

        try {
            loadModel();   // Load mô hình YOLO
            runInference();  // Chạy suy luận
        } catch (Exception e) {
            Log.e(TAG, "Error in onCreate", e);
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    // Tải mô hình YOLO từ tệp tflite trong thư mục assets
    private void loadModel() throws IOException {
        try {
            MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(this, WEIGHT_PATH);
            Interpreter.Options options = new Interpreter.Options();
            tflite = new Interpreter(modelBuffer, options);
            Log.d(TAG, "Model loaded successfully");
        } catch (IOException e) {
            Log.e(TAG, "Error loading model", e);
            Toast.makeText(this, "Failed to load model: " + e.getMessage(), Toast.LENGTH_LONG).show();
            throw e;
        }
    }

    // Phương thức thực hiện suy luận YOLO
    private void runInference() {
        try {
            Log.d(TAG, "Starting inference...");

            Bitmap frame = getDummyFrame(); // Lấy frame dummy để thử nghiệm
            Bitmap resized = Bitmap.createScaledBitmap(frame, 640, 640, true); // Resize ảnh về kích thước 640x640

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

            // Lấy thông tin về shape của đầu vào và đầu ra
            int[] inputShape = tflite.getInputTensor(0).shape();
            Log.d(TAG, "Model input shape: " + java.util.Arrays.toString(inputShape));

            int[] outputShape = tflite.getOutputTensor(0).shape();
            Log.d(TAG, "Model output shape: " + java.util.Arrays.toString(outputShape));

            // Kiểm tra nếu shape của đầu ra là [4, 5, 8400] như mong đợi
            if (outputShape.length == 3 && outputShape[0] == 4 && outputShape[1] == 5 && outputShape[2] == 8400) {
                TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputShape, org.tensorflow.lite.DataType.FLOAT32);

                Log.d(TAG, "Running inference...");

                // Chạy suy luận với TensorFlow Lite
                tflite.run(inputBuffer, outputBuffer.getBuffer().rewind());
                Log.d(TAG, "Inference completed.");

                float[] outputArray = outputBuffer.getFloatArray();
                Log.d(TAG, "Output size: " + outputArray.length);

                // Kiểm tra kích thước mảng đầu ra, nếu đúng sẽ tiếp tục xử lý
                if (outputArray.length == 168000) {
                    Log.d(TAG, "Output array is correctly sized: " + outputArray.length);

                    // Reshape lại mảng đầu ra thành mảng 3D [4, 5, 8400]
                    float[][][] reshapedOutput = new float[4][5][8400];
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 5; j++) {
                            for (int k = 0; k < 8400; k++) {
                                int index = (i * 5 * 8400) + (j * 8400) + k;
                                reshapedOutput[i][j][k] = outputArray[index];
                            }
                        }
                    }

                    // Bây giờ bạn có thể sử dụng reshapedOutput với dạng chính xác
                    drawResult(resized, reshapedOutput);  // Vẽ kết quả lên ảnh
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

    // Phương thức vẽ bounding box lên ảnh
    private void drawResult(Bitmap frame, float[][][] outputArray) {
        try {
            Bitmap mutable = frame.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutable);
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5f);

            Paint textPaint = new Paint();
            textPaint.setColor(Color.RED);
            textPaint.setTextSize(40);
            textPaint.setStyle(Paint.Style.FILL);

            // Biến lưu thông tin box có score cao nhất
            float highestScore = 0;
            float bestX1 = 0, bestY1 = 0, bestX2 = 0, bestY2 = 0;
            boolean foundBox = false;

            // Tìm box có confidence cao nhất từ tất cả các batch và detections
            for (int batchIndex = 0; batchIndex < 4; batchIndex++) {
                for (int i = 0; i < 8400; i++) {
                    float confidence = outputArray[batchIndex][4][i];

                    // Cập nhật nếu tìm thấy confidence cao hơn
                    if (confidence > highestScore) {
                        highestScore = confidence;

                        // Lấy tọa độ của box
                        float x_center = outputArray[batchIndex][0][i] * frame.getWidth();
                        float y_center = outputArray[batchIndex][1][i] * frame.getHeight();
                        float width = outputArray[batchIndex][2][i] * frame.getWidth();
                        float height = outputArray[batchIndex][3][i] * frame.getHeight();

                        // Chuyển đổi từ 'xywh' sang 'xyxy'
                        bestX1 = x_center - width / 2;
                        bestY1 = y_center - height / 2;
                        bestX2 = x_center + width / 2;
                        bestY2 = y_center + height / 2;

                        foundBox = true;
                    }
                }
            }

            // Vẽ box có confidence cao nhất nếu tìm thấy
            if (foundBox) {
                // Vẽ bounding box
                canvas.drawRect(bestX1, bestY1, bestX2, bestY2, paint);

                // Vẽ tên đối tượng và confidence
                String label = String.format("Ball: %.2f", highestScore);
                canvas.drawText(label, bestX1, bestY1 - 10, textPaint);

                Log.d(TAG, "Drew best box at: " + bestX1 + ", " + bestY1 + ", " +
                        bestX2 + ", " + bestY2 + " with confidence: " + highestScore);
            } else {
                Log.d(TAG, "No boxes detected");
            }

            resultView.setImageBitmap(mutable); // Hiển thị kết quả lên ImageView
        } catch (Exception e) {
            Log.e(TAG, "Error drawing result", e);
            e.printStackTrace();
        }
    }

    // Phương thức tạo frame dummy cho thử nghiệm
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
