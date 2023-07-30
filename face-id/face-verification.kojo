// scroll down to the bottom of the file 
// for code to play with

cleari()
clearOutput()

import java.io.File
import java.awt.image.BufferedImage
import java.util.HashMap

import scala.util.Using

import org.tensorflow.types.TFloat32
import org.tensorflow.{ SavedModelBundle, Tensor }

import org.bytedeco.javacv._
import org.bytedeco.opencv.global.opencv_core.CV_32F
import org.bytedeco.opencv.global.opencv_dnn.{ readNetFromCaffe, blobFromImage }
import org.bytedeco.opencv.opencv_core.{ Rect, Mat, Size, Scalar, Point }
import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.opencv.global.opencv_imgproc.rectangle
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.global.opencv_imgcodecs.imread

import net.kogics.kojo.tensorutil.imgToTensorF
import net.kogics.kojo.nst.removeAlphaChannel
import net.kogics.kojo.util.Utils

val scriptDir = Utils.kojoCtx.baseDir
val confidenceThreshold = 0.5
val fdModelConfiguration = new File(s"$scriptDir/face_detection_model/deploy.prototxt")
val fdModelBinary = new File(s"$scriptDir/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
val inWidth = 300
val inHeight = 300
val inScaleFactor = 1.0
val meanVal = new Scalar(104.0, 177.0, 123.0, 128)

val markerColor = new Scalar(0, 255, 255, 0)

val vggFaceSavedModel = s"$scriptDir/vggface_resnet50_model"

require(
    fdModelConfiguration.exists(),
    s"Cannot find FD model configuration: ${fdModelConfiguration.getCanonicalPath}")

require(
    fdModelBinary.exists(),
    s"Cannot find FD model file: ${fdModelConfiguration.getCanonicalPath}")

require(
    new File(vggFaceSavedModel).exists,
    s"Cannot find VGGFace model file: ${fdModelConfiguration.getCanonicalPath}")

val faceDetectionNet = readNetFromCaffe(fdModelConfiguration.getCanonicalPath, fdModelBinary.getCanonicalPath)
val vggfaceNet = SavedModelBundle.load(vggFaceSavedModel)

var pic: Picture = _
var lastFrameTime = epochTimeMillis
@volatile var currImageMat: Mat = _
@volatile var currFaces: Seq[Rect] = _
def detectSequence(grabber: FrameGrabber): Unit = {
    val delay = 1000.0 / fps
    grabber.start()
    try {
        var frame = grabber.grab()
        while (frame != null) {
            frame = grabber.grab()
            val currTime = epochTimeMillis
            if (currTime - lastFrameTime > delay) {
                val imageMat = Java2DFrameUtils.toMat(frame)
                if (imageMat != null) { // sometimes the first few frames are empty so we ignore them
                    val faces = locateAndMarkFaces(imageMat)
                    val pic2 = Picture.image(Java2DFrameUtils.toBufferedImage(imageMat))
                    centerPic(pic2, imageMat.size(1), imageMat.size(0))
                    pic2.draw()
                    currImageMat = imageMat
                    currFaces = faces
                    if (pic != null) {
                        pic.erase()
                    }
                    pic = pic2
                    lastFrameTime = currTime
                }
            }
        }
    }
    catch {
        case _ => // eat up interruption
    }
    finally {
        grabber.stop()
        scriptDone()
    }
}

def centerPic(pic: Picture, w: Int, h: Int) {
    pic.translate(-w / 2, -h / 2)
}

def locateAndMarkFaces(image: Mat): Seq[Rect] = {
    // We will need to scale results for display on the input image, we need its width and height
    val imageWidth = image.size(1)
    val imageHeight = image.size(0)

    // Convert image to format suitable for using with the net
    val inputBlob = blobFromImage(
        image, inScaleFactor, new Size(inWidth, inHeight), meanVal, false, false, CV_32F)

    // Set the network input
    faceDetectionNet.setInput(inputBlob)

    // Make forward pass, compute output
    val t0 = epochTime
    val detections = faceDetectionNet.forward()
    val t1 = epochTime

    // Decode detected face locations
    val di = detections.createIndexer().asInstanceOf[FloatIndexer]
    val faceRegions =
        for {
            i <- 0 until detections.size(2)
            confidence = di.get(0, 0, i, 2)
            if (confidence > confidenceThreshold)
        } yield {
            val x1 = (di.get(0, 0, i, 3) * imageWidth).toInt
            val y1 = (di.get(0, 0, i, 4) * imageHeight).toInt
            val x2 = (di.get(0, 0, i, 5) * imageWidth).toInt
            val y2 = (di.get(0, 0, i, 6) * imageHeight).toInt
            new Rect(new Point(x1, y1), new Point(x2, y2))
        }

    for (rect <- faceRegions) {
        rectangle(image, rect, markerColor)
    }

    faceRegions
}

def faceEmbedding(image: BufferedImage): Array[Float] = {
    faceEmbedding(Java2DFrameUtils.toMat(image))
}

def faceEmbedding(image: Mat): Array[Float] =
    Using.Manager { use =>
        val src = Java2DFrameUtils.toBufferedImage(image)
        val args = new HashMap[String, Tensor]()
        val inputTensor = imgToTensorF(removeAlphaChannel(src, white))
        args.put("input_1", inputTensor)
        val out = use(vggfaceNet.call(args).get("global_average_pooling2d").get.asInstanceOf[TFloat32])
        val data = out.asRawTensor.data.asFloats
        val ab = new ArrayBuffer[Float]
        for (n <- 0 until 2048) ab.append(data.getFloat(n))
        ab.toArray
    }.get

def extractAndResizeFace(imageMat: Mat, rect: Rect): Mat = {
    val faceMat = new Mat(imageMat, rect)
    resize(faceMat, faceMat, new Size(224, 224))
    faceMat
}

def checkFace(imageMat: Mat, faces: Seq[Rect]): Boolean = {
    if (currFaces.size == 1) {
        val faceMat = extractAndResizeFace(imageMat, faces(0))
        val emb = faceEmbedding(faceMat)
        val dist = distance(emb, learnedEmbedding)
        println(s"Distance - $dist")
        if (dist < similarityThreshold) true else false
    }
    else {
        println("Verification is done only if there is one face on the screen")
        false
    }

}

def distance(a: Array[Float], b: Array[Float]): Float =
    math.sqrt(a.zip(b).map { case (x1, x2) =>
        math.pow(x1 - x2, 2)
    }.sum).toFloat

def scriptDone() {
    println("Closing Nets...")
    vggfaceNet.close()
    faceDetectionNet.close()
}

var learnedEmbedding: Array[Float] = _

val btnWidth = 100
val gap = 10
val iRadius = 30

def button(label: String): Picture = {
    picStackCentered(
        Picture.rectangle(btnWidth, iRadius * 2).withFillColor(cm.lightBlue).withPenColor(gray),
        Picture.text(label).withPenColor(black)
    )
}

val cb = canvasBounds
val learnButton = button("Learn")
val verifyButton = button("Verify")
val learnStatusIndicator = Picture.circle(iRadius).withPenColor(gray)
val verifyStatusIndicator = Picture.circle(iRadius).withPenColor(gray)

val panel =
    picRowCentered(
        learnStatusIndicator, Picture.hgap(gap),
        learnButton, Picture.hgap(gap), verifyButton,
        Picture.hgap(gap), verifyStatusIndicator
    )

draw(panel)
panel.setPosition(
    cb.x + (cb.width - panel.bounds.width) / 2 + iRadius,
    cb.y + 50)

val grabber = new OpenCVFrameGrabber(0)
Utils.runAsyncMonitored {
    detectSequence(grabber)
}

// -----------------------------
// Tweak stuff below as desired

val similarityThreshold = 90
val fps = 5 

learnButton.onMouseClick { (x, y) =>
    if (currFaces.size == 1) {
        learnStatusIndicator.setFillColor(yellow)
        Utils.runAsyncQueued {
            val faceMat = extractAndResizeFace(currImageMat, currFaces(0))
            learnedEmbedding = faceEmbedding(faceMat)
            learnStatusIndicator.setFillColor(orange)
        }
    }
    else {
        println("There should be only one face on the screen for Learning")
        learnStatusIndicator.setFillColor(noColor)
        learnedEmbedding = null
    }

}

verifyButton.onMouseClick { (x, y) =>
    val good = checkFace(currImageMat, currFaces)
    if (good) {
        verifyStatusIndicator.setFillColor(green)
    }
    else {
        verifyStatusIndicator.setFillColor(red)
    }
    Utils.schedule(2) {
        verifyStatusIndicator.setFillColor(noColor)
    }
}
