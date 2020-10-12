package recognition;

import java.util.function.UnaryOperator;

public abstract class NetworkFunction {
    public UnaryOperator<Double> function = null;
    public UnaryOperator<Double> derivative = null;

    public double function(double param) {
        return function.apply(param);
    }

    public double derivative(double param) {
        return derivative.apply(param);
    }
}

