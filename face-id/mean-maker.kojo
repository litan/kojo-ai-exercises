cleari()
clearOutput()

def button(label: String): Picture = {
    picStackCentered(
        Picture.rectangle(200, 50).withFillColor(cm.lightBlue).withPenColor(gray),
        Picture.text(label).withPenColor(black)
    )
}

val btn = button("Click to add number")
drawCentered(btn)

var numbers = ArrayBuffer.empty[Array[Float]]

def mean(fes: ArrayBuffer[Array[Float]]): Array[Float] = {
    val elen = fes(0).length
    // create empty array of length elen
    var result = new Array[Float](elen)
    for (ctr <- 0 until elen) {
        var sum = 0f
        for (idx <- 0 until fes.length) {
            val n = fes(idx)(ctr)
            sum += n
        }
        result(ctr) = sum / fes.length
    }
    result
}

btn.onMouseClick { (x, y) =>
    val num = random(10, 20)
    val numA = Array[Float](num)
    numbers.append(numA)
    val mn = mean(numbers)
    // arrays don't print well; so converting to 
    // arraybuffer for printing
    println(s"Added seq - ${numA.toBuffer}")
    println(s"New mean is - ${mn.toBuffer}")
}
