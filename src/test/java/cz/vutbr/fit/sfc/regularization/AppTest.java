package cz.vutbr.fit.sfc.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class AppTest
{
    @Test
    public void testMatrixMult()
    {
        INDArray first = Nd4j.create(new float[] { 1, 5, 2, 3, 1, 7 }, new int[] { 3, 2 });
        INDArray second = Nd4j.create(new float[] { 1, 2, 3, 7, 5, 2, 8, 1 }, new int[] { 2, 4 });
        INDArray expected = Nd4j.create(new float[] { 26, 12, 43, 12, 17, 10, 30, 17, 36, 16, 59, 14 }, new int[] { 3, 4 });
        INDArray actual = first.mmul(second);
        assertEquals(actual, expected);
    }
}
