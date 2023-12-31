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
val fdConfidenceThreshold = 0.5
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
    s"Cannot find FD model file: ${fdModelBinary.getCanonicalPath}")

require(
    new File(vggFaceSavedModel).exists,
    s"Cannot find VGGFace model dir: ${vggFaceSavedModel}")

val faceDetectionNet = readNetFromCaffe(fdModelConfiguration.getCanonicalPath, fdModelBinary.getCanonicalPath)
val vggfaceNet = SavedModelBundle.load(vggFaceSavedModel)

@volatile var videoFramePic: Picture = _
@volatile var lastFrameTime = epochTimeMillis
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
                    val vfPic2 = Picture.image(Java2DFrameUtils.toBufferedImage(imageMat))
                    centerPic(vfPic2, imageMat.size(1), imageMat.size(0))
                    vfPic2.draw()
                    currImageMat = imageMat
                    currFaces = faces
                    if (videoFramePic != null) {
                        videoFramePic.erase()
                    }
                    videoFramePic = vfPic2
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
    val detections = faceDetectionNet.forward()

    // Decode detected face locations
    val di = detections.createIndexer().asInstanceOf[FloatIndexer]
    val faceRegions =
        for {
            i <- 0 until detections.size(2)
            confidence = di.get(0, 0, i, 2)
            if (confidence > fdConfidenceThreshold)
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

def scaler(r: Float, g: Float, b: Float): (Float, Float, Float) = {
    val switchToBgr = true
    val (bD, gD, rD) = (91.4953f, 103.8827f, 131.0912f)
    val (r2, g2, b2) = (r - rD, g - gD, b - bD)
    if (switchToBgr) (b2, g2, r2) else (r2, g2, b2)
}

def faceEmbedding(image: Mat): Array[Float] =
    Using.Manager { use =>
        val src = Java2DFrameUtils.toBufferedImage(image)
        val args = new HashMap[String, Tensor]()
        val inputTensor = imgToTensorF(removeAlphaChannel(src), Some(scaler _))
        args.put("input_1", inputTensor)
        val out = use(vggfaceNet.call(args).get("global_average_pooling2d").get.asInstanceOf[TFloat32])
        val data = out.asRawTensor.data.asFloats
        val ab = new ArrayBuffer[Float]
        for (n <- 0 until data.size.toInt) ab.append(data.getFloat(n))
        ab.toArray
    }.get

def extractAndResizeFace(imageMat: Mat, rect: Rect): Mat = {
    val faceMat = new Mat(imageMat, rect)
    resize(faceMat, faceMat, new Size(224, 224))
    faceMat
}

def scriptDone() {
    println("Closing Nets...")
    vggfaceNet.close()
    faceDetectionNet.close()
}

val btnWidth = 100
val gap = 10
val iRadius = 30

def button(label: String): Picture = {
    picStackCentered(
        Picture.rectangle(btnWidth, iRadius * 2).withFillColor(cm.lightBlue).withPenColor(gray),
        Picture.text(label).withPenColor(black)
    )
}

// -----------------------------
// Tweak stuff below as desired
// Some ideas:
// Tweak indicator colors
// Tweak threshold and explore false positives, false negatives, etc
// make buttons inactive while work is happening
// Learn an "average" face for better performance

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
    cb.y + iRadius + 5)

val similarityThreshold = 90
val fps = 5
var learnedEmbedding: Array[Float] = _

def distance(a: Array[Float], b: Array[Float]): Float = {
    math.sqrt(
        a.zip(b).map { case (x1, x2) =>
            math.pow(x1 - x2, 2)
        }.sum
    ).toFloat
}

def distance2(a: Array[Float], b: Array[Float]): Float = {
    require(a.length == b.length)
    var sumSqdiff = 0.0
    for (idx <- 0 until a.length) {
        val an = a(idx); val bn = b(idx)
        val sqDiff = math.pow(an - bn, 2)
        sumSqdiff += sqDiff
    }
    math.sqrt(sumSqdiff).toFloat
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

learnButton.onMouseClick { (x, y) =>
    if (currFaces.size == 1) {
        println("Learning Face")
        learnStatusIndicator.setFillColor(cm.purple)
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
    if (learnedEmbedding != null) {
        val good = checkFace(currImageMat, currFaces)
        if (good) {
            verifyStatusIndicator.setFillColor(green)
        }
        else {
            verifyStatusIndicator.setFillColor(red)
        }
        Utils.schedule(1) {
            verifyStatusIndicator.setFillColor(noColor)
        }
    }
    else {
        println("First Learn, then Verify!")
    }
}

val grabber = new OpenCVFrameGrabber(0)
Utils.runAsyncMonitored {
    detectSequence(grabber)
}
