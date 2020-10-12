package recognition;

import java.io.Serializable;

public class NeuralLayer implements Serializable {
//    Neuron[] neurons;
    transient NetworkFunction networkFunction;
    Matrix weight;
    transient Matrix delta;
    transient Matrix A;
    transient Matrix dZ;
    transient Matrix error;

    NeuralLayer(int sizeInput, int numberOfNeurons, NetworkFunction networkFunction) {
//        neurons = new Neuron[numberOfNeurons];
        this.networkFunction = networkFunction;
        weight = new Matrix(sizeInput, numberOfNeurons);
        Matrix.fillRandom(weight);
    }

    @Override
    public String toString() {
        String sb = "weight={" + weight.toString() + "}\n" +
                "delta={" + (delta == null ? "null" : delta.toString()) + "}\n" +
                "A={" + (A == null ? "null" : A.toString()) + "}\n" +
                "dZ={" + (dZ == null ? "null" : dZ.toString()) + "}\n";
        return sb;
    }

    Matrix forward(Matrix x) {
//        System.out.println("weight:\n" + weight);
//        System.out.println("x:\n" + x);
        Matrix z = x.multiply(weight);
//        System.out.println("z:\n" + z);
        A = z.func(networkFunction.function);
//        System.out.println("A:\n" + A);
        dZ = z.func(networkFunction.derivative);
//        System.out.println("dZ:\n" + dZ);
        return A;
    }

    Matrix backward(Matrix y, NeuralLayer right) {
        // if output layer
        if (right == null) {
//            System.out.println("A:\n" + A);
//            System.out.println("y:\n" + y);
            error = A.minus(y);
//            System.out.println("error:\n" + error);
//            System.out.println("dZ:\n" + dZ);
            delta = error.dot(dZ);
//            System.out.println("delta:\n" + delta);
        } else {
//            System.out.println("right.delta:\n" + right.delta);
//            System.out.println("right.weight.transpose():\n" + right.weight.transpose());
//            System.out.println("dZ:\n" + dZ);
            delta = right.delta.multiply(right.weight.transpose()).dot(dZ);
        }
        return delta;
    }

    double update(double learningRate, Matrix leftA) {
//        System.out.println("leftA:\n" + leftA);
//        System.out.println("delta:\n" + delta);
        Matrix ad = leftA.transpose().multiply(delta);
//        System.out.println("ad:\n" + ad);
//        System.out.println("ad.multiply(learningRate):\n" + ad.multiply(learningRate));
//        System.out.println("weight before update:\n" + weight);
        weight = weight.subtract(ad.multiply(learningRate));
//        System.out.println("weight after update:\n" + weight);
        double res = 0.0;
        for (int i = 0; i < ad.getRows(); i++) {
            for (int j = 0; j < ad.getColumns(); j++) {
                res += Math.abs(ad.getAt(i, j));
            }
        }
        return res;
    }



//    public Matrix guess(Matrix weights, Matrix input) {
//        // weights No x Ni
//        // input Ni x 1
//        // result No x 1
//        // add bias to input
//        input = input.addRow(new double[]{1.0});
//        Matrix result = weights.multiply(input);
//        for (int i = 0; i < result.getRows(); i++) {
//            neurons[i] = new Neuron(result.getAt(i, 0));
//        }
//        return result.func(networkFunction::function);
//    }
//
//    public Matrix partialDerivative() {
//        Matrix result = new Matrix(1, neurons.length);
//        for (int i = 0; i < neurons.length; i++) {
//            result.setAt(0, i, neurons[i].value);
//        }
//        return result.func(networkFunction::derivative);
//    }
//
//    public Matrix getOutput() {
//        Matrix result = new Matrix(1, neurons.length + 1);
//        for (int i = 0; i < neurons.length; i++) {
//            result.setAt(0, i, neurons[i].value);
//        }
//        result.setAt(0, neurons.length, 1.0); // bias output = 1
//        return result;
//    }
//
//    public Matrix getResult() {
//        Matrix result = new Matrix(1, neurons.length + 1);
//        for (int i = 0; i < neurons.length; i++) {
//            result.setAt(0, i, neurons[i].value);
//        }
//        result = result.func(networkFunction::function);
//        result.setAt(0, neurons.length, 1.0); // bias output = 1
//        return result;
//    }

    public static void main(String[] args) {
        NeuralLayer nl = new NeuralLayer(3, 4, new Sigmoid());
        NeuralLayer nlRight = new NeuralLayer(4, 5, new Sigmoid());
        Matrix x = new Matrix(new double[][]{{1, 2, 3}});
        System.out.println("first layer");
        Matrix res = nl.forward(x);
        System.out.println("res:\n" + res);
        System.out.println("second layer");
        res = nlRight.forward(res);
        Matrix y = new Matrix(new double[][]{{1, 0, 0, 1, 0}});
        Matrix delta = nlRight.backward(y, null);
        System.out.println("inner layer");
        delta = nl.backward(y, nlRight);
        System.out.println("update");
        nl.update(0.5, x);
        x = nl.A;
        nlRight.update(0.5, x);
    }
}
