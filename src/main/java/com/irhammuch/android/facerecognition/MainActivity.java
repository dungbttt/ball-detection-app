package com.irhammuch.android.facerecognition;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;
import org.tensorflow.lite.support.common.FileUtil;

public class MainActivity extends AppCompatActivity {

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
        } catch (IOException e) {
            Log.e("TFLite", "Model load failed", e);
        }
    }

    private void loadModel() throws IOException {
        MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(this, "yolov8_3.tflite");
        tflite = new Interpreter(modelBuffer);
    }

    private void runInference() {
        // Replace with actual video frame extraction logic
        Bitmap frame = getDummyFrame();
        Bitmap resized = Bitmap.createScaledBitmap(frame, 640, 640, true);

        TensorImage image = TensorImage.fromBitmap(resized);
        TensorBuffer input = image.getTensorBuffer();

        // Assuming output is of shape [1, 2] for x, y prediction
        TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, 2}, org.tensorflow.lite.DataType.FLOAT32);

        tflite.run(input.getBuffer(), output.getBuffer().rewind());

        float[] prediction = output.getFloatArray();
        drawResult(frame, prediction);
    }

    private void drawResult(Bitmap frame, float[] prediction) {
        Bitmap mutable = frame.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutable);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.FILL);
        paint.setTextSize(40);

        float x = prediction[0] * frame.getWidth();
        float y = prediction[1] * frame.getHeight();

        canvas.drawCircle(x, y, 10, paint);
        canvas.drawText("Ball", x + 15, y + 15, paint);

        resultView.setImageBitmap(mutable);
    }

    private Bitmap getDummyFrame() {
        // Dummy method, replace with real frame (e.g. from video or CameraX)
        return Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
    }
}
