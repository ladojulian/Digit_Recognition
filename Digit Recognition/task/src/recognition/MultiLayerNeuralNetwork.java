package recognition;

import java.io.*;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MultiLayerNeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;
    static Logger logger = Logger.getLogger("MultiLayerNeuralNetwork");

    NeuralLayer[] layers;
    transient NetworkFunction networkFunction;
//    Matrix[] weights;

    /**
     * @param layersDesc - [0] - size of input
     *                     [1..n-2] - sizes of inner layers
     *                     [n-1] - size of output layer
     */
    MultiLayerNeuralNetwork(int[] layersDesc) {
        networkFunction = new Sigmoid();
        layers = new NeuralLayer[layersDesc.length - 1];
//        weights = new Matrix[layersDesc.length - 1];
        for (int i = 1; i < layersDesc.length - 1; i++) {
//            // plus bias (size of layer plus one more row in every matrix)
//            weights[i] = new Matrix(layersDesc[i + 1], layersDesc[i] + 1);
            layers[i - 1] = new NeuralLayer(layersDesc[i - 1] + 1, layersDesc[i] + 1, networkFunction);
        }
        layers[layers.length - 1] = new NeuralLayer(
                layersDesc[layersDesc.length - 2] + 1,
                layersDesc[layersDesc.length - 1], networkFunction);
    }

    public void learn(Matrix idealInputs, Matrix idealOutputs, int numberOfEpochs, double learningRate, double maxError) {
        Random random = new Random();
//        System.out.println("idealInputs:\n" + idealInputs);
        idealInputs = idealInputs.addColumn(1.0, 0);
//        System.out.println("idealInputs:\n" + idealInputs);
//        System.out.println("idealOutputs:\n" + idealOutputs);
        double error = 0.0;
        for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
            error = 0.0;
            for (int i = 0; i < idealInputs.getRows(); i++) {
//            int i = random.nextInt(idealInputs.getRows());
                Matrix a = idealInputs.getRow(i);
                guess(a);
//                System.out.println(a);
//                System.out.println(idealOutputs.getRow(i));

                Matrix delta = layers[layers.length - 1].backward(idealOutputs.getRow(i), null);
                for (int j = layers.length - 2; j >= 0; j--) {
                    delta = layers[j].backward(delta, layers[j + 1]);
                }

//                Matrix a = idealInputs.getRow(i);
                for (NeuralLayer layer : layers) {
                    error += layer.update(learningRate, a);
                    a = layer.A;
                }
            }
            if (epoch % 100 == 0) {
                System.out.println("epoch #" + epoch + " error = " + error);
            }
            if (error < maxError) {
                System.out.println("epoch #" + epoch + " error = " + error);
                break;
            }
        }
        System.out.println("error: " + error);
    }

    public void learn2(Matrix idealInputs, Matrix idealOutputs, int numberOfEpochs, double learningRate, double maxError) {
        idealInputs = idealInputs.addColumn(1.0, 0);
        final int batchSize = idealInputs.getRows() / 70;
        int batchStart = 0;
        Matrix partInputs = idealInputs.getMatrixPart(batchStart, batchSize);
        Matrix partOutputs = idealOutputs.getMatrixPart(batchStart, batchSize);
        double error = 0.0;
        for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
            error = 0.0;

            guess(partInputs);

            Matrix delta = layers[layers.length - 1].backward(partOutputs, null);
            for (int j = layers.length - 2; j >= 0; j--) {
                delta = layers[j].backward(delta, layers[j + 1]);
            }

            Matrix a = partInputs;
            for (NeuralLayer layer : layers) {
                error += layer.update(learningRate, a);
                a = layer.A;
            }
            if (epoch % 10 == 0) {
                showInputOutputs(partInputs.getRow(0), partOutputs.getRow(0), layers[layers.length-1].A.getRow(0));
                System.out.println("epoch #" + epoch + " error = " + error);
            }
            if (error < maxError) {
                System.out.println("epoch #" + epoch + " error = " + error);
//                break;
            }
            batchStart += batchSize;
            if (batchStart + batchSize >= idealInputs.getRows()) {
                batchStart = 0;
            }
            partInputs = idealInputs.getMatrixPart(batchStart, batchStart + batchSize);
            partOutputs = idealOutputs.getMatrixPart(batchStart, batchStart + batchSize);
        }
        System.out.println("error: " + error);
    }

    private void showInputOutputs(Matrix input, Matrix idealOutput, Matrix realOutput) {
        System.out.println("input:");
        for (int i = 0; i < input.getColumns(); i++) {
            int num = (int) (input.getAt(0, i) * 9);
            System.out.print(num > 0 ? num : " ");
            if (i % 28 == 0) {
                System.out.println();
            }
        }
        System.out.println("ideal output:");
        System.out.println(idealOutput);
        System.out.println("real output:");
        System.out.println(realOutput);
    }


    /*    public void learn(Matrix idealInputs, Matrix idealOutputs, boolean randomWeights) {
            if (randomWeights) {
                setRandomWeights();
            }
    //        for (Matrix weight : weights) {
    //            System.out.println(weight);
    //        }
    //        Matrix results = new Matrix(idealInputs.getRows(), idealOutputs.getRows());
            for (int i = 0; i < idealInputs.getRows(); i++) {
                Matrix result = guess(idealInputs.getRow(i).transpose());
                result = result.transpose();
                for (int j = 0; j < layers.length; j++) {
                    System.out.println(layers[j]);
                }
                System.out.println("result:\n" + result);
                System.out.println("ideal out:\n" + idealOutputs.getRow(i));
                Matrix errors = idealOutputs.getRow(i).subtract(result);
                System.out.println("errors:\n" + errors);
                Matrix sigma = errors.dot(layers[layers.length - 1].partialDerivative());
                System.out.println("pd:\n" + layers[layers.length - 1].partialDerivative());
                System.out.println("sigma:\n" + sigma);

                System.out.println("layers-2 result:\n" + layers[layers.length - 2].getResult());
    //            System.out.println("weights[weights.length - 1]:\n" + weights[weights.length - 1]);
                Matrix dW = sigma.transpose().multiply(layers[layers.length - 2].getResult());
                System.out.println("dW:\n" + dW);

    /*            System.out.println("errors:\n" + errors);
                errors = errors.func(x -> x * x / 2);
                System.out.println("squared errors:\n" + errors);
                Matrix totalErrors = errors.multiply(Matrix.getMatrix(errors.getColumns(), 1, 1));
                System.out.println("total errors:\n" + totalErrors);
                Matrix pdErrorsToOut = result.subtract(idealOutputs.getRow(i));
                System.out.println("pdErrorsToOut:\n" + pdErrorsToOut);
                Matrix pdResult = result.func(networkFunction::derivative);
                System.out.println("pdResult:\n" + pdResult);
                Matrix pdWeight = layers[layers.length - 2].getOutput();
                System.out.println("pdWeight:\n" + pdWeight);
                Matrix outputDeltas = pdErrorsToOut
                        .dot(pdResult).transpose()
                        .multiply(pdWeight);
                System.out.println("output deltas:\n" + outputDeltas);
                System.out.println("weights[weights.length - 1]:\n" + weights[weights.length - 1]);
                Matrix wDelta = outputDeltas.multiply(0.5);
                System.out.println("wDelta:\n" + wDelta);
                Matrix newWeights = weights[weights.length - 1].subtract(wDelta);
                System.out.println("newWeights:\n" + newWeights);

                Matrix dEo = pdErrorsToOut.transpose().multiply(pdResult);
                System.out.println("dEo:\n" + dEo);
                System.out.println("weights[weights.length - 1]:\n" + weights[weights.length - 1]);
                System.out.println(dEo.multiply(weights[weights.length - 1]));
    //            weights[weights.length - 1] = newWeights;
    //            System.out.println("weights:\n" + weights[layers.length-1]);


            }
        }
     */
    public Matrix predict(Matrix input) {
        input = input.addColumn(1.0, 0);
        return guess(input);
    }

    private Matrix guess(Matrix input) {
        for (NeuralLayer layer : layers) {
            input = layer.forward(input);
        }
        return input;
    }

//    public void setWeights(Matrix[] weights) {
//        this.weights = weights;
//    }

//    private void setRandomWeights() {
//        Random random = new Random();
//        for (int i = 0; i < weights.length; i++) {
//            for (int j = 0; j < weights[i].getRows(); j++) {
//                for (int k = 0; k < weights[i].getColumns(); k++) {
//                    weights[i].setAt(j, k, random.nextGaussian());
//                }
//            }
//        }
//    }

    public void serialize(String fileName) throws IOException {
        FileOutputStream fos = new FileOutputStream(fileName);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        ObjectOutputStream oos = new ObjectOutputStream(bos);
        oos.writeObject(this);
//        for (NeuralLayer layer : layers) {
//            logger.log(Level.INFO, layer.toString());
//        }
        logger.log(Level.INFO, "serialize to file: " + fileName);
        oos.close();
    }

    public void deserialize(String fileName) throws IOException, ClassNotFoundException {
        logger.log(Level.INFO, "deserialize from file: " + fileName);
        FileInputStream fis = new FileInputStream(fileName);
        BufferedInputStream bis = new BufferedInputStream(fis);
        ObjectInputStream ois = new ObjectInputStream(bis);
        MultiLayerNeuralNetwork nn = (MultiLayerNeuralNetwork) ois.readObject();
        layers = nn.layers;
        for (int i = 0; i < layers.length; i++) {
            layers[i].networkFunction = networkFunction;
//            logger.log(Level.INFO, layers[i].toString());
        }
        ois.close();
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        MultiLayerNeuralNetwork neuralNetwork = new MultiLayerNeuralNetwork(new int[]{2, 3, 4, 1});

//        neuralNetwork.learn(NumberRecognition.idealInputs, NumberRecognition.idealOutputs);

//        neuralNetwork.deserialize("c:\\temp\\mlnetwork.neu");

        Matrix x = new Matrix(new double[][]{
                {0, 0}, // 1
                {1, 0},
                {0, 1},
                {1, 1}});
        Matrix y = new Matrix(new double[][]{
                {0},
                {1},
                {1},
                {0},
        });
        neuralNetwork.learn2(x, y, 20000, 0.1, 0.01);

        for (int i = 0; i < x.getRows(); i++) {
            System.out.println("guess:" + x.getRow(i) + neuralNetwork.predict(x.getRow(i)));
        }
    }
}
