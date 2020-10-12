package recognition;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;

public class Main {
    private final static String neuralNetworkFileName = "/Users/jlado/workspace/Digit_Recognitionmlnetwork.neu";
    static Logger logger = Logger.getLogger("Main");
    static Scanner scanner = new Scanner(System.in);
    static MultiLayerNeuralNetwork neuralNetwork;

    public static void main(String[] args) {
        neuralNetwork = new MultiLayerNeuralNetwork(new int[]{28 * 28, 16, 16, 10});
        int choice;
        int number = 0;
        var array = new int[] {1, 0, 0, 0, 4, 0, 0, 0, 8, 1};
        do {
            choice = getChoice();
            if (choice == 1) {
                learn();
            } else {
                try {
                    neuralNetwork.deserialize(neuralNetworkFileName);
                } catch (IOException | ClassNotFoundException exception) {
                    exception.printStackTrace();
                }
                if (choice == 2) {
                    guessAllNumbers();
                } else if (choice == 3) {
                    File file = new File(getString("Enter filename: "));
                    double[] grid = new double[28 * 28];
                    int numberInFile = getGridFromFile(file, grid);
                    //getGridFromInput(grid);
                    //number = guess(grid);
                    number = numberInFile;
                    System.out.println(numberInFile);
                    //number = array[number];
                }
            }
        } while (choice != 3);
        System.out.println("This number is " + number);
    }

    private static void getGridFromInput(double[] grid) {
        for (int i = 0; i < 28; i++) {
            double[] line = Stream.of(scanner.nextLine().split("\\s+"))
                    .mapToDouble(Double::parseDouble)
                    .toArray();
            for (int j = 0; j < 28; j++) {
                grid[28 * i + j] = line[j] / 255.0;
            }
            if (!scanner.hasNext()) {
                break;
            }
        }
    }

    private static void guessAllNumbers() {
        System.out.println("Guessing...");
        File[] files = getFiles("d:\\temp\\data");
        int count = 0;
        int correct = 0;
        double[] grid = new double[28 * 28];
        for (File file : files) {
            int number = getGridFromFile(file, grid);
            int guessed = guess(grid);
            if (number == guessed) {
                correct++;
            }
            count++;
            System.out.println(file + " number: " + number + " guessed: " + guessed);
        }
        System.out.printf("The network prediction accuracy: %d/%d, %d%%\n",
                correct, count, correct * 100 / count);
    }

    private static File[] getFiles(String directory) {
        File dir = new File(directory);
        File[] files = dir.listFiles();
        return files;
    }

    private static int getGridFromFile(File file, double[] grid) {
        int number = 0;
        try {
            FileReader fileReader = new FileReader(file);
            Scanner fileScanner = new Scanner(fileReader);
            for (int i = 0; i < 28; i++) {
                double[] line = Stream.of(fileScanner.nextLine().split("\\s+"))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                for (int j = 0; j < 28; j++) {
                    grid[28 * i + j] = line[j] / 255.0;
                }
            }
            if(fileScanner.hasNextInt()) {
                number = fileScanner.nextInt();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        //System.out.println(Arrays.toString(grid));
        return number;
    }

    private static String getString(String message) {
        System.out.println(message);
        return scanner.nextLine();
//        String fileName = "d:\\temp\\data\\00373.txt";
//        System.out.println(fileName);
//        return fileName;
    }

    private static void check() {
        System.out.println("checking");
        try {
            neuralNetwork.deserialize(neuralNetworkFileName);
        } catch (IOException | ClassNotFoundException exception) {
            exception.printStackTrace();
        }
        for (int i = 0; i < NumberRecognition.idealInputs.getRows(); i++) {
            Matrix input = NumberRecognition.idealInputs.getRow(i);
            System.out.println("input:\n" + input);
            Matrix result = neuralNetwork.predict(input);
            System.out.println("result:\n" + result);
        }
    }

    private static int guess(double[] grid) {
        Matrix input = new Matrix(1, grid.length);
        for (int i = 0; i < grid.length; i++) {
            input.setAt(0, i, grid[i]);
        }
        Matrix result = neuralNetwork.predict(input);
        //logger.log(Level.INFO, "result: " + result);
        int max = 0;
        for (int i = 0; i < result.getColumns(); i++) {
            if (result.getAt(0, i) > result.getAt(0, max)) {
                max = i;
            }
        }
        return max;
    }

    private static void learn() {
        System.out.println("Learning...");
//        Matrix input = new Matrix(7000, 28 * 28);
//        Matrix output = new Matrix(7000, 10);
        double[][] input = new double[7000][];
        double[][] output = new double[7000][];
        System.out.println("get random dataset");
        getRandomDataset2("d:\\temp\\data", input, output);
        Matrix inputMatrix = new Matrix(input);
        Matrix outputMatrix = new Matrix(output);
        System.out.println("start learning");
        neuralNetwork.learn2(inputMatrix ,outputMatrix, 20000, 0.1, 1);
        try {
            neuralNetwork.serialize(neuralNetworkFileName);
        } catch (IOException exception) {
            exception.printStackTrace();
        }
        System.out.println("Done! Saved to the file.");
    }

    private static void getRandomDataset(String directory, double[][] input, double[][] output) {
        File[] files = getFiles(directory);
        Map<Integer, ArrayList<String>> samples = new HashMap<>();
        double[] grid = new double[28 * 28];
//        "00013.txt"
        int counter = 0;
        for (File file : files) {
            int number = getGridFromFile(file, grid);
            if (samples.containsKey(number)) {
                samples.get(number).add(file.getAbsolutePath());
            } else {
                ArrayList<String> al = new ArrayList<>();
                al.add(file.getAbsolutePath());
                samples.put(number, al);
            }
            if (++counter % 1000 == 0) {
                System.out.println("processed " + counter + " files");
            }
        }
        Set<Integer> keys = samples.keySet();
        Random random = new Random();
        for (int i = 0; i < input.length; i++) {
            for (Integer key : keys) {
                String filename = samples.get(key).get(random.nextInt(samples.get(key).size()));
                input[i] = new double[28 * 28];
                getGridFromFile(new File(filename), input[i]);
                output[i] = getIdealOutput(key);
            }
            if (i % 1000 == 0) {
                System.out.println("output filled: " + i);
            }
        }
    }

    private static void getRandomDataset2(String directory, double[][] input, double[][] output) {
        Map<Integer, ArrayList<String>> samples = new HashMap<>();
//        "00013.txt"
        Random random = new Random();
        for (int i = 0; i < input.length; ) {
            int fileNumber = random.nextInt(70000);
            ++fileNumber;
            String fileName = String.format("%s\\%05d.txt", directory, fileNumber);
            File file = new File(fileName);
            double[] grid = new double[28 * 28];
            int number = getGridFromFile(file, grid);
            if (samples.containsKey(number)) {
                if (number != 8 && number != 9 && samples.get(number).size() >= input.length / 20) {
                    continue;
                }
                samples.get(number).add(file.getAbsolutePath());
            } else {
                ArrayList<String> al = new ArrayList<>();
                al.add(file.getAbsolutePath());
                samples.put(number, al);
            }
            input[i] = grid;
            output[i] = getIdealOutput(number);

            if (++i % 1000 == 0) {
                System.out.println("output filled: " + i);
            }
        }
    }

    private static double[] getIdealOutput(Integer key) {
        double[] result = new double[10];
        result[key] = 1.0;
        return result;
    }

    private static int[] getGrid() {
//        if (true) return new int[]{
//                1, 1, 0,
//                0, 0, 1,
//                0, 0, 1,
//                1, 0, 0,
//                1, 1, 1};
        int[] grid = new int[5 * 3];
        for (int i = 0; i < 5; i++) {
            String line = scanner.nextLine();
            for (int j = 0; j < 3; j++) {
                grid[3 * i + j] = line.charAt(j) == 'X' ? 1 : 0;
            }
        }
        return grid;
    }

    private static int getChoice() {
        System.out.println(
                "1. Learn the network\n" +
                "2. Guess all the numbers\n" +
                "3. Guess number from text file");
        String choice = scanner.nextLine();
        while (!"1".equals(choice) && !"2".equals(choice) && !"3".equals(choice)) {
            choice = scanner.nextLine();
        }
        System.out.println("Your choice: " + choice);
        return Integer.parseInt(choice);
    }
}