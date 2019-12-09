package cz.vutbr.fit.sfc.regularization.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ActivationFunction
{
    public abstract INDArray f(INDArray i);
    public abstract INDArray df(INDArray i);
}
