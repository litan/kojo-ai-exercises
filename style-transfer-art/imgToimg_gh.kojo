// #include /nst.kojo

cleari()
clearOutput()

val baseDir = kojoCtx.baseDir

val alpha = 1f

val style = new NeuralStyleFilter(
    s"$baseDir/nst_model_gh/",
    s"$baseDir/style/wave.png",
    alpha
)

val content = Picture.image(
    s"$baseDir/content/golden_gate_bridge.png"
)

val pic = content.withEffect(style)

drawCentered(pic)