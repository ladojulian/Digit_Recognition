package recognition;

public class Sigmoid extends NetworkFunction {
    Sigmoid() {
        function = x -> 1 / (1 + Math.exp(-x));
        derivative = x -> function(x) * (1 - function(x));
    }

    public static void main(String[] args) {
        Sigmoid s = new Sigmoid();
        System.out.println(s.function(1));
        System.out.println(s.function(10));
        System.out.println(s.function(-1));
        System.out.println(s.function(-10));
        System.out.println(s.derivative(1));
    }
}
