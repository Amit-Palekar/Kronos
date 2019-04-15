package com.example.amitpalekar.face;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.KalmanFilter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 * @author Amit Palekar
 * This program is an android activity which tracks a face over discrete time intervals
 * and estimates the position of the face at all points using a Kalman Filter.
 * @version 1.0
 * Please direct any questions to: amit12.palekar@gmail.com
 */

public class MainActivity extends Activity implements CvCameraViewListener {
    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;
    private KalmanFilter kf;
    private double lastX, lastY;
    private int frameCount = 0;
    private Mat measurement;
    private final int DELTA_FRAMES = 1;
    //set this to the number of frames over which you want to update the filter

    private void initializeOpenCVDependencies() {
        try {
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);


            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }

        // And we are ready to go
        openCvCameraView.enableView();
    }

    /*
      Creates UI and resets the Kalman Filter
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        System.loadLibrary("opencv_java3");

        openCvCameraView = new JavaCameraView(this, -1);
        openCvCameraView.enableFpsMeter();
        initializeOpenCVDependencies();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(openCvCameraView);
        openCvCameraView.setCvCameraViewListener(this);
        resetFilter();
    }

    //Initializes the Kalman filter with a default state
    public void resetFilter() {
        measurement = new Mat(4, 1, CvType.CV_32F);

        kf = new KalmanFilter(5, 4, 0, CvType.CV_32F);

        /*  Transition Matrix:
            [   1,  0,  dt, 0,  0   ]
            [   0,  1,  0,  dt, 0   ]
            [   0,  0,  1,  0,  0   ]
            [   0,  0,  0,  1,  0   ]
            [   0   0,  0,  0,  1   ]
         */

        Mat tMatrix = new Mat(5, 5, CvType.CV_32F);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (i == j)
                    tMatrix.put(i, j, 1);
                else
                    tMatrix.put(i, j, 0);
            }
        }

        tMatrix.put(0, 2, DELTA_FRAMES);
        tMatrix.put(1, 3, DELTA_FRAMES);

        kf.set_transitionMatrix(tMatrix);

        Core.setIdentity(kf.get_measurementMatrix());
        Core.setIdentity(kf.get_processNoiseCov(), Scalar.all(2));
        Core.setIdentity(kf.get_measurementNoiseCov(), Scalar.all(100));
        Core.setIdentity(kf.get_errorCovPost(), Scalar.all(1));
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);
        absoluteFaceSize = (int) (height * 0.3); // The faces will be a 20% of the height of the screen
    }

    @Override
    public void onCameraViewStopped() { }

    //Detect faces in each frame and update the filter at regular intervals
    @Override
    public Mat onCameraFrame(Mat frame) {
        if (frameCount % DELTA_FRAMES == 0) {
            Imgproc.cvtColor(frame, grayscaleImage, Imgproc.COLOR_BGR2GRAY); //Create a grayscale image
            Imgproc.equalizeHist(grayscaleImage, grayscaleImage);
            Imgproc.GaussianBlur(grayscaleImage, grayscaleImage, new Size(5, 5), 2); //Gaussian Blur

            MatOfRect faces = new MatOfRect();
            MatOfInt num = new MatOfInt(1);

            // Use the classifier to detect faces
            if (cascadeClassifier != null) {
                cascadeClassifier.detectMultiScale2(grayscaleImage, faces, num, 1.1, 5, 1,
                        new Size(absoluteFaceSize, absoluteFaceSize), new Size());
            }

            // If there are any faces found, draw a rectangle around it
            Rect[] facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++) {
                Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);

                Point center = new Point();
                center.x = facesArray[i].x + (facesArray[i].br().x - facesArray[i].x)/2;
                center.y = facesArray[i].y + (facesArray[i].br().y - facesArray[i].y)/2;
                //Imgproc.circle(frame, center, absoluteFaceSize/2, new Scalar(0, 255, 0, 255), 3);

                measurement.put(0, 0, center.x);
                measurement.put(1, 0, center.y);
                measurement.put(2, 0, center.x - lastX);
                measurement.put(3, 0, center.y - lastY);

                //For changing delta frames
                /*int df = frameCount - lastFrame;
                lastFrame = frameCount;

                Mat transitionMatrix = kf.get_transitionMatrix();
                transitionMatrix.put(0, 2, df);
                transitionMatrix.put(1, 3, df);
                kf.set_transitionMatrix(transitionMatrix);*/

                //First predict to update the internal statePre variable
                Mat prediction = kf.predict();

                //Update step
                kf.correct(measurement);

                System.out.println("KALMAN PREDICTION");
                for (int k = 0; k < 5; k++) {
                    double[] points = prediction.get(k, 0);
                    for (double point : points)
                        System.out.print(point + ", ");
                    System.out.println();
                }
                lastX = center.x;
                lastY = center.y;
                frameCount = 0;
            }
        }

        Mat prediction = kf.predict();
        double[] a = prediction.get(0, 0);
        double[] b = prediction.get(1, 0);
        Imgproc.circle(frame, new Point(a[0], b[0]), absoluteFaceSize, new Scalar(255, 0, 0, 0), 3);

        frameCount++;

        return frame;
    }
}
