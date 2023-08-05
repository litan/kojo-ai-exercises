clearOutput()

import java.util.ArrayList

import scala.util.Using

import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.op.core.Placeholder

def getFloat(n: Tensor): Float = {
    n.asInstanceOf[TFloat32].getFloat()
}

def gradient(tf: Ops, y: Operand[TFloat32], x: Operand[TFloat32]): Operand[_] = {
    val a = new ArrayList[Operand[TFloat32]](1)
    a.add(x)
    tf.gradients(y, a).iterator().next()
}

val graph = new Graph()
val tf = Ops.create(graph)

// define symbolic computation graph
val xSym = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.scalar))
val ySym = tf.math.mul(
    tf.constant(3f),
    tf.math.pow(xSym, tf.constant(2f))
)
val gradYSym = gradient(tf, ySym, xSym)

// run computation graph in a session; feed in actual data - fetch actual results
val xActual = TFloat32.scalarOf(10)
Using.Manager { use =>
    val session = use(new Session(graph))
    val result = use(session.runner()
        .feed(xSym, xActual)
        .fetch(ySym)
        .fetch(gradYSym)
        .run())

    val yResult = result.get(0)
    val yGradResult = result.get(1)
    println(s"y = ${getFloat(yResult)}, dy/dx = ${getFloat(yGradResult)}")
}

graph.close()