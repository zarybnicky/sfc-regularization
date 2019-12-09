package cz.vutbr.fit.sfc.regularization;

import java.util.ArrayList;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import cz.vutbr.fit.sfc.regularization.activation.ActivationFunction;

import static org.nd4j.linalg.ops.transforms.Transforms.sign;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;

public class Network implements Cloneable
{
    public boolean l1;
    public boolean l2;
    int[] sizes;
    ArrayList<INDArray> weights = new ArrayList<>();
    ArrayList<INDArray> biases = new ArrayList<>();
    ActivationFunction activation;

    public Network(ActivationFunction activation, boolean l1, boolean l2, int... sizes)
    {
        this(activation, sizes, l1, l2);
    }
    public Network(ActivationFunction activation, int[] sizes, boolean l1, boolean l2)
    {
        this.l1 = l1;
        this.l2 = l2;
        this.sizes = sizes;
        this.activation = activation;
        for (int i = 1; i < sizes.length; i++) {
            biases.add(Nd4j.randn(sizes[i], 1));
        }
        for (int i = 1; i < sizes.length; i++) {
            INDArray w = Nd4j.randn(sizes[i], sizes[i - 1]);
            weights.add(w.div(Math.sqrt(sizes[i - 1])));
        }
    }

    public INDArray forward(INDArray x)
    {
        // Calculate feed-forward
        for (int i = 0; i < biases.size(); i++) {
            x = activation.f(weights.get(i).mmul(x).add(biases.get(i)));
        }
        return x;
    }

    public int test(DataSet ds)
    {
        // Calculate number of mis-classified samples, assumes one-hot encoding
        // of label and final network layer.
        int errors = 0;
        for (DataSet s : ds.asList()) {
            if (!forward(s.getFeatures().transpose()).argMax().equals(s.getLabels().argMax())) {
                errors++;
            }
        }
        return errors;
    }

    public double learnBatch(DataSet ds, double alpha, double lambda, int n)
    {
        INDArray[] nablaB = biases.stream().map(b -> Nd4j.zeros(b.shape())).toArray(INDArray[]::new);
        INDArray[] nablaW = weights.stream().map(w -> Nd4j.zeros(w.shape())).toArray(INDArray[]::new);
        double loss = 0;

        for (DataSet sample : ds.asList()) {
            INDArray x = sample.getFeatures().transpose();
            INDArray y = sample.getLabels().transpose();

            // Feed-forward
            ArrayList<INDArray> zs = new ArrayList<>();
            ArrayList<INDArray> as = new ArrayList<>();
            as.add(x);
            for (int i = 0; i < biases.size(); i++) {
                x = weights.get(i).mmul(x).add(biases.get(i));
                zs.add(x);
                x = activation.f(x);
                as.add(x);
            }

            // Backpropagation step
            INDArray delta = as.get(as.size() - 1).sub(y);
            nablaB[biases.size() - 1].addi(delta);
            nablaW[weights.size() - 1].addi(delta.mmul(as.get(as.size() - 2).transpose()));
            for (int i = biases.size() - 2; i >= 0; i--){
                delta = weights.get(i + 1).transpose().mmul(delta).muli(activation.df(zs.get(i)));
                nablaB[i].addi(delta);
                nablaW[i].addi(delta.mmul(as.get(i).transpose()));
            }

            // Mean squares loss function
            loss += pow(as.get(as.size() - 1).sub(y), 2).sum().muli(0.5).getDouble(0);
        }

        for (int i = 0; i < biases.size(); i++) {
            // Regularization
            if (l2) {
                weights.get(i).muli(1 - alpha * lambda / n);
            }
            if (l1) {
                weights.get(i).addi(sign(weights.get(i)).muli(alpha * lambda / n));
            }

            // Weight update
            biases.get(i).subi(nablaB[i].mul(alpha / ds.getFeatures().length()));
            weights.get(i).subi(nablaW[i].mul(alpha / ds.getFeatures().length()));
        }
        return loss;
    }

    public Network clone() throws CloneNotSupportedException
    {
        Network n = new Network(activation, sizes, l1, l2);
        n.weights = weights.stream().map(w -> w.dup()).collect(Collectors.toCollection(ArrayList::new));
        n.biases = biases.stream().map(b -> b.dup()).collect(Collectors.toCollection(ArrayList::new));
        return n;
    }
}
