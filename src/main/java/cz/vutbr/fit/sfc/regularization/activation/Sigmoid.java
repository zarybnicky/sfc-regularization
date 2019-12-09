package cz.vutbr.fit.sfc.regularization.activation;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoidDerivative;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Sigmoid implements ActivationFunction
{
    public INDArray f(INDArray i)
    {
        return sigmoid(i);
    }

    public INDArray df(INDArray i)
    {
        return sigmoidDerivative(i);
    }
}
