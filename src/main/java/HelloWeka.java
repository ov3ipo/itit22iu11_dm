import weka.classifiers.trees.J48;

public class HelloWeka {
    public static void main(String[] args) {
        DataLoader loader = new DataLoader("data/sample.csv");
        loader.loadData();

        loader.saveToArff("processed_data.arff");
    }
}
