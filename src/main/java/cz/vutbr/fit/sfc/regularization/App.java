package cz.vutbr.fit.sfc.regularization;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Stream;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

import cz.vutbr.fit.sfc.regularization.activation.Sigmoid;

/**
 * Zadání 11, obtížnost = 0.8, Demonstrace učení BP - regularizace
 */
public class App
{
    public static void main(String[] args) throws IOException, CloneNotSupportedException
    {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        // Parsing of the input file - fairly robust, should work for most other
        // example datasets.
        File file = new File("data/mehrotra/B1-Iris.dta");
        final double[][] featureArr;
        final ArrayList<Double> labelsRaw = new ArrayList<>();
        try (Stream<String> linesStream = Files.lines(file.toPath())) {
            featureArr = linesStream.filter(x -> !x.isEmpty()).map(line -> {
                double[] words = Arrays.stream(line.split(" "))
                    .filter(x -> !x.isEmpty())
                    .mapToDouble(Double::parseDouble)
                    .toArray();
                labelsRaw.add(words[words.length - 1]);
                return Arrays.copyOfRange(words, 0, words.length - 2);
            }).toArray(double[][]::new);
        }
        int maxLabel = (int) Math.round(labelsRaw.stream().max(Double::compare).orElse(0.) + 1);
        final double[][] labelArr = labelsRaw.stream().map(label -> {
                double[] l = new double[maxLabel];
            l[(int) Math.round(label)] = 1;
            return l;
        }).toArray(double[][]::new);

        // Layer size setting
        int[] sizes = new int[] { featureArr[0].length, 10, labelArr[0].length };

        // Learning parameters
        double alpha = 0.01; // Learning rate
        double lambda = 5; // Regularization rate

        Network netNoReg = new Network(new Sigmoid(), sizes, false, false);
        Network netL1Reg = netNoReg.clone();
        netL1Reg.l1 = true;
        Network netL2Reg = netNoReg.clone();
        netL2Reg.l2 = true;

        // Split data into training and test set
        double trainingSetPercent = 0.8;
        DataSet ds = new DataSet(Nd4j.create(featureArr), Nd4j.create(labelArr));
        SplitTestAndTrain stt = ds.splitTestAndTrain(trainingSetPercent);
        DataSet dsTrain = stt.getTrain();
        DataSet dsTest = stt.getTest();

        // ... and start learning
        int n = ds.asList().size();
        for (int i = 0; i < 1000; i++) {
            double lossNoReg = netNoReg.learnBatch(dsTrain, alpha, lambda, n);
            double lossL1Reg = netL1Reg.learnBatch(dsTrain, alpha, lambda, n);
            double lossL2Reg = netL2Reg.learnBatch(dsTrain, alpha, lambda, n);
            if (i % 10 == 0) {
                // System.out.printf("%s|%s|%s|%s\n", i, lossNoReg, lossL1Reg, lossL2Reg);
                System.out.printf("Epoch %s, errors: \t", i);
                System.out.printf("No regularization: %s/%s (loss %s)\t",
                                  netNoReg.test(dsTest), dsTest.asList().size(), lossNoReg);
                System.out.printf("L1 regularization: %s/%s (loss %s)\t",
                                  netL1Reg.test(dsTest), dsTest.asList().size(), lossL1Reg);
                System.out.printf("L2 regularization: %s/%s (loss %s)\n",
                                  netL2Reg.test(dsTest), dsTest.asList().size(), lossL2Reg);
            }
        }
    }
}
