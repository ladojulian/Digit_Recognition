package recognition;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.UnaryOperator;

public class Matrix implements Serializable {
    private final int rows;
    private final int columns;
    private final double[][] data;

    Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        data = new double[rows][columns];
    }
    Matrix(double[][] data) {
        this.data = data;
        this.rows = data.length;
        this.columns = data[0].length;
    }

    public Matrix(Matrix other) {
        rows = other.rows;
        columns = other.columns;
        data = new double[rows][];
        for (int i = 0; i < rows; i++) {
            data[i] = Arrays.copyOf(other.data[i], other.data[i].length);
        }
    }

    static public Matrix getMatrix(int rows, int columns, int element) {
        assert rows > 0 && columns > 0;
        Matrix result = new Matrix(rows, columns);
        for (int i = 0; i < result.getRows(); i++) {
            Arrays.fill(result.data[i], element);
        }
        return result;
    }

    public static void fillRandom(Matrix matrix) {
        Random random = new Random();
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.columns; j++) {
                matrix.data[i][j] = random.nextGaussian();
            }
        }
    }

    public double getAt(int row, int column) {
        assert row < data.length && column < data[row].length;
        return data[row][column];
    }

    public void setAt(int row, int column, double value) {
        assert row < data.length && column < data[row].length;
        data[row][column] = value;
    }

    public void setRow(int rowSet, Matrix other, int rowGet) {
        assert this.data[rowSet].length == other.data[rowGet].length;
        this.data[rowSet] = Arrays.copyOf(other.data[rowGet], other.data[rowGet].length);
    }

    public Matrix addRow(double value) {
        double[] row = new double[columns];
        Arrays.fill(row, value);
        return addRow(row);
    }

    public Matrix addRow(double[] row) {
        return addRow(row, getRows());
    }

    public Matrix addRow(double[] row, int position) {
        assert position <= rows && row.length == columns;
        Matrix result = new Matrix(getRows() + 1, getColumns());
        for (int i = 0; i < position; i++) {
            result.data[i] = Arrays.copyOf(data[i], data[i].length);
        }
        result.data[position] = Arrays.copyOf(row, row.length);
        for (int i = position; i < getRows(); i++) {
            result.data[i + 1] = Arrays.copyOf(data[i], data[i].length);
        }
        return result;
    }

    public Matrix getRow(int row) {
        assert row > 0 && row < row;
        Matrix result = new Matrix(1, getColumns());
        for (int i = 0; i < getColumns(); i++) {
            result.setAt(0, i, getAt(row, i));
        }
        return result;
    }

    public Matrix getColumn(int column) {
        assert column < columns;
        Matrix result = new Matrix(getRows(), 1);
        for (int i = 0; i < getRows(); i++) {
            result.setAt(i, 0, getAt(i, column));
        }
        return result;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public Matrix multiply(Matrix other) {
        assert columns == other.rows;
        int rows = this.getRows();
        int columns = other.getColumns();
        Matrix result = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                double sum = 0.0;
                for (int r = 0; r < this.getColumns(); r++) {
                    sum += this.getAt(i, r) * other.getAt(r, j);
                }
                result.setAt(i, j, sum);
            }
        }
        return result;
    }

    public Matrix multiply(double number) {
        Matrix result = new Matrix(getRows(), getColumns());
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getColumns(); j++) {
                result.setAt(i, j, getAt(i, j) * number);
            }
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getColumns(); j++) {
                sb.append(String.format("%.3f ", getAt(i, j)));
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public Matrix transpose() {
        double[][] result = new double[columns][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[j][i] = data[i][j];
            }
        }
        return new Matrix(result);
    }

    public Matrix func(UnaryOperator<Double> f) {
        Matrix result = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = f.apply(data[i][j]);
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other) {
        Matrix result = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    public Matrix add(Matrix other) {
        Matrix result = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    public Matrix dot(Matrix other) {
        assert rows == other.rows && columns == other.columns;
        Matrix result = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    public Matrix minus(Matrix other) {
        Matrix result = new Matrix(rows, columns);
        if (other.columns == 1) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    result.data[i][j] = data[i][j] - other.data[i][0];
                }
            }
        } else if (other.rows == 1) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    result.data[i][j] = data[i][j] - other.data[0][j];
                }
            }
        } else if (rows == other.rows && columns == other.columns) {
            result = subtract(other);
        } else {
            System.out.println("error in minus()");
        }
        return result;
    }

    public Matrix addColumn(double value, int position) {
        Matrix result = new Matrix(rows, columns + 1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < position; j++) {
                result.data[i][j] = data[i][j];
            }
            result.data[i][position] = value;
            for (int j = position; j < columns; j++) {
                result.data[i][j + 1] = data[i][j];
            }
        }
        return result;
    }

    public Matrix getMatrixPart(int startRow, int endRow) {
        assert startRow < endRow && startRow < rows && endRow < rows;
        Matrix result = new Matrix(endRow - startRow, columns);
        for (int i = 0; i < result.rows; i++) {
            result.data[i] = Arrays.copyOf(data[startRow + i], data[startRow + i].length);
        }
        return result;
    }

    public static void main(String[] args) {
        double[][] data = {
                {1,2,3,4},
                {5,6,7,8},
                {9,0,1,2}
        };
        double[][] ones = {{1},{1},{1},{1}};
        double[][] data2 = {
                {1,2,3,4},
                {5,6,7,8},
                {9,0,1,2},
                {3,4,5,6}
        };
        Matrix m2 = new Matrix(data2);
        System.out.println(m2.getMatrixPart(1,3));
        Matrix m = new Matrix(data);
        System.out.println(m.toString());
        System.out.println(m.addColumn(999.0, 0));
        Matrix o = new Matrix(ones);
        System.out.println(m.multiply(o).toString());
        System.out.println(m.addRow(new double[]{1, 2, 3, 4}));
        Matrix p = getMatrix(10, 20, 1);
        System.out.println(p);
        p = getMatrix(1, 5, 1);
        System.out.println(p);
        p = getMatrix(5, 1, 1);
        System.out.println(p);
        /*Matrix n = new Matrix(data2);
        System.out.println(n.toString());
        System.out.println(m.multiply(n).toString());
        System.out.println(m.multiply(2));
        System.out.println(n.multiply(0));*/
    }
}
