// #include /nst.kojo

cleari()
clearOutput()

val baseDir = kojoCtx.baseDir

val style = new NeuralStyleFilter2(
    s"$baseDir/nst_model_th/",
    s"$baseDir/style/wave_256.png",
)

val content = Picture.image(
    s"$baseDir/content/golden_gate_bridge.png"
)

val pic = content.withEffect(style)

drawCentered(pic)