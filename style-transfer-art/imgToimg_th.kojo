// #include /nst.kojo

cleari()
clearOutput()

val baseDir = kojoCtx.baseDir
val modelDir = s"$baseDir/nst_model_th/"
val styleImage = s"$baseDir/style/wave_256.png"
val contentImage = s"$baseDir/content/golden_gate_bridge.png"

def checkExists(filename: String, desc: String) {
    import java.io.File
    require(new File(filename).exists, s"$desc does not exist")
}

checkExists(modelDir, "Model directory")
checkExists(styleImage, "Style image")
checkExists(contentImage, "Content image")

val style = new NeuralStyleFilter2(modelDir, styleImage)
val content = Picture.image(contentImage)

val pic = content.withEffect(style)
drawCentered(pic)
