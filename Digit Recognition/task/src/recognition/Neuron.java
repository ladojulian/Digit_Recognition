package recognition;

public class Neuron {
    double value;

    public Neuron(double value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return String.format("%.3f", value);
    }
}
