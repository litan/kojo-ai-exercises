clearOutput()

import java.util.ArrayList

import scala.util.Using

import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TInt32
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.op.core.Placeholder

def getInt(n: Tensor): Int = {
    //    n.asRawTensor().data().asInts().getInt(0)
    n.asInstanceOf[TInt32].getInt()
}

def gradient(tf: Ops, y: Operand[TInt32], x: Operand[TInt32]): Operand[_] = {
    val a = new ArrayList[Operand[TInt32]](1)
    a.add(x)
    tf.gradients(y, a).iterator().next()
}

val graph = new Graph()
val tf = Ops.create(graph)

//val xSym = tf.variable(Shape.scalar, classOf[TInt32])
val xSym = tf.placeholder(classOf[TInt32], Placeholder.shape(Shape.scalar))
val ySym = tf.math.mul(
    tf.constant(3),
    tf.math.mul(xSym, xSym),
)
val gradYSym = gradient(tf, ySym, xSym)

val xActual = TInt32.scalarOf(10)
Using(new Session(graph)) { session =>
    val result = session.runner()
        .feed(xSym, xActual)
        .fetch(ySym)
        .fetch(gradYSym)
        .run()

    val yResult = result.get(0)
    val yGradResult = result.get(1)
    println(s"y = ${getInt(yResult)}, dy/dx = ${getInt(yGradResult)}")
}

