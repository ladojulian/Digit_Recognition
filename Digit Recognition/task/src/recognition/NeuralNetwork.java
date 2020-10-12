package recognition;

import java.io.*;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;
    static Logger logger = Logger.getLogger("NeuralNetwork");

    private double[][] weights;

    private void showArray(double[][] array, String message) {
        System.out.println(message);
        for (double[] doubles : array) {
            if (doubles == null) {
                return;
            }
            for (double aDouble : doubles) {
                System.out.printf("%.3f ", aDouble);
            }
            System.out.println();
        }
    }

    private void showNeurons(Neuron[] neurons, String message) {
        System.out.print(message);
        for (Neuron neuron : neurons) {
            if (neuron == null) {
                return;
            }
            System.out.printf("%.2f ", neuron.value);
        }
        System.out.println();
    }

    private void showNeurons(Neuron[][] neurons, String message) {
        System.out.print(message);
        for (Neuron[] neuron : neurons) {
            for (Neuron n : neuron) {
                if (neuron == null) {
                    return;
                }
                System.out.printf("%.2f ", n.value);
            }
            System.out.println();
        }
    }

    public Neuron[] guess(Neuron[] input) {
        return guess(input, weights);
    }

    private Neuron[] guess(Neuron[] input, double[][] weights) {
//        showNeurons(input, "input: ");
        Neuron[] output = new Neuron[weights.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = new Neuron(0.0);
            assert input.length == (weights[i].length - 1);
            for (int j = 0; j < input.length; j++) {
                output[i].value += input[j].value * weights[i][j];
//                System.out.printf("%.2f*%.2f+",weights[i][j], input[j].value);
            }
            output[i].value += weights[i][input.length];
//            System.out.printf("%.2f*1=%.2f\n",weights[i][input.length], output[i].value);
            output[i].value = 1.0 / (1.0 + Math.exp(-output[i].value));
//            output[i].value = Math.floor(output[i].value * 100d) / 100d;
        }
        return output;
    }

    public void learn(Neuron[][] idealInput, Neuron[][] idealOutput) {
        double[] myGaussian = {
                0.21,  0.32, -0.92, 0.03,
               -0.34, -0.21,  0.93, 0.49,
                0.31,  0.01, -0.79, 0.61,
                0.73, -0.47, -0.44, 0.39};

        double[][] weights = new double[idealInput.length][];
        Random random = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = new double[idealInput[i].length + 1]; // plus bias
            for (int j = 0; j < weights[i].length; j++) {
//                weights[i][j] = random.nextGaussian();
                weights[i][j] = myGaussian[j];
            }
        }
        showArray(weights, "initial weights:");

        boolean enough = false;
        int generation = 0;
        while (!enough) {

//            showArray(weights, "initial weights:");
            Neuron[][] output = new Neuron[idealInput.length][];
            for (int i = 0; i < idealInput.length; i++) {
                output[i] = guess(idealInput[i], weights);
            }
//            for (int i = 0; i < output.length; i++) {
//                showNeurons(output[i], "Neurons for " + i + ": ");
//            }

            double averageMean = 0.0;
            // Для каждого выходного нейрона oi считамем dW
            for (int i = 0; i < output.length; i++) {
                // Для oi идеальная цифра - i
                // Δw (aj, oi) = η ∗ aj ∗ (oi ideal - oi)
                for (int j = 0; j < idealInput[0].length; j++) {
                    double dWmean = 0.0;
//                    System.out.printf("for neuron o%d, for a%d\n", i, j);
                    for (int inpNum = 0; inpNum < idealInput.length; inpNum++) {
                        double element = 0.5 * idealInput[inpNum][j].value * (idealOutput[inpNum][i].value - output[inpNum][i].value);
//                        System.out.printf("Δw (a%d, o%d) = η ∗ a%d ∗ (o%d ideal - o%d) = %.3f\n", j, i, j, i, i, element);
                        dWmean += element;
                    }
                    dWmean /= idealInput.length;
//                    System.out.printf("Δwmean(a%d, o%d) = %.3f\n", j, i, dWmean);
                    weights[i][j] += dWmean;
                    averageMean += Math.abs(dWmean);
                }
            }
/*
            double averageMean = 0.0;
            for (int i = 0; i < output.length; i++) { // i = [0..9]
                double[][] dW = new double[output[i].length][];
                for (int j = 0; j < dW.length; j++) { // j = [0..9]
                    dW[j] = new double[idealInput[j].length];
                    for (int k = 0; k < dW[j].length; k++) { // k = [0..15)
//                        dW[j][k] = 0.5 * idealInput[j][k].value * (idealOutput[i][j].value - output[i][j].value);
                        dW[j][k] = 0.5 * idealInput[j][k].value * (idealInput[i][k].value - output[i][j].value);
                    }
                }
                showArray(dW, "dW for " + i + ": ");
//                System.out.print("Mean values:");
                double[] mean = new double[dW[0].length];
                for (int j = 0; j < dW[0].length; j++) {
                    for (int k = 0; k < dW.length; k++) {
                        mean[j] += dW[k][j];
                    }
                    mean[j] = mean[j] / dW.length;
//                    System.out.printf(" %.2f", mean[j]);
                }
//                System.out.println("");
                // update weight
                for (int j = 0; j < mean.length; j++) {
//                    weights[i][j] += mean[j];
                    averageMean += Math.abs(mean[j]);
                }
//                showArray(weights, "weights updated after " + i);
            }
            */
            if (averageMean < 0.001) {
                enough = true;
            }
            generation++;
            if (generation % 100000 == 0) {
                System.out.printf("gen: %d, averageMean = %.5f\n", generation, averageMean);
                showArray(weights, "weights: ");
                showNeurons(output, "neurons: ");
            }
        }
        showArray(weights, "end weights: ");
        this.weights = weights;
/*
        double[][] output = new double[ideal.length][];
        double[][][] dW = new double[ideal.length][][];
        double[][] meanDW = new double[ideal.length][];
        for (int i = 0; i < output.length; i++) {
            output[i] = new double[ideal.length];
            dW[i] = new double[ideal.length][];
            meanDW[i] = new double[weights[i].length - 1];
            System.out.println("o" + i);
            for (int j = 0; j < output[i].length; j++) {
                for (int k = 0; k < weights[j].length - 1; k++) {
                    output[i][j] += weights[j][k] * ideal[j][k].value;
                    System.out.print(weights[j][k] + "*" + ideal[j][k].value + " + ");
                }
                // plus bias
                output[i][j] += weights[j][weights[j].length - 1];
                System.out.print(weights[j][weights[j].length - 1] + "*1=" + output[i][j]);
                System.out.print(", S(" + output[i][j]);
                output[i][j] = 1.0 / (1 + Math.exp(-output[i][j]));
                System.out.println(")=" + output[i][j]);

                dW[i][j] = new double[weights[j].length - 1];
                for (int k = 0; k < dW[i][j].length; k++) {
                    dW[i][j][k] = 0.5 * ideal[j][k].value * (ideal[i][k].value - output[i][j]);
//                    System.out.println("dW(" + j + "," + k + "o" + i + ")="
//                            + ideal[j][k].value + "*(" + ideal[i][k].value + "-" + output[i][j] + ")="
//                            + dW[i][j][k]);
                    meanDW[i][k] += dW[i][j][k];
                }
            }
            showArray(dW[i], ("dW[" + i + "]"));
            showArray(meanDW, ("meanDW[" + i + "]"));
            for (int j = 0; j < meanDW[i].length; j++) {
                meanDW[i][j] /= meanDW.length;
            }
            showArray(meanDW, ("meanDW[" + i + "] averaged"));
        }
        showArray(output, "output");*/
    }

    public double[][] getWeights() {
        return weights;
    }

    public void serialize(String fileName) throws IOException {
        FileOutputStream fos = new FileOutputStream(fileName);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        ObjectOutputStream oos = new ObjectOutputStream(bos);
        oos.writeObject(this);
        oos.close();
    }

    public void deserialize(String fileName) throws IOException, ClassNotFoundException {
        logger.log(Level.INFO, "deserialize from file: " + fileName);
        FileInputStream fis = new FileInputStream(fileName);
        BufferedInputStream bis = new BufferedInputStream(fis);
        ObjectInputStream ois = new ObjectInputStream(bis);
        NeuralNetwork nn = (NeuralNetwork) ois.readObject();
        weights = nn.weights;
        ois.close();
    }

    public Neuron[] guess(int[] grid) {
        Neuron[] input = new Neuron[grid.length];
        for (int i = 0; i < input.length; i++) {
            input[i] = new Neuron(grid[i]);
        }
        return guess(input);
    }
}
