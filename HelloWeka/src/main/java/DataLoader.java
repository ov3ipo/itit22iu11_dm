import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import java.io.File;

public class DataLoader {
    private Instances data;
    private String dataPath;
    private boolean isDataLoaded = false;

    // Constructor
    public DataLoader(String dataPath) {
        this.dataPath = dataPath;
    }

    /**
     * Loads data from CSV file
     * @return true if loading is successful
     */
    public boolean loadData() {
        try {
            System.out.println("Loading data from: " + dataPath);
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(dataPath));
            data = loader.getDataSet();
            isDataLoaded = true;

            // Print initial data statistics
            printDataStats();

            return true;
        } catch (Exception e) {
            System.err.println("Error loading data: " + e.getMessage());
            return false;
        }
    }

    /**
     * Clean the data by removing missing values for specific attributes and standardizing numeric attributes
     * @param attributeIndices array of attribute indices to check for missing values
     */
    public void cleanData(int[] attributeIndices) {
        if (!isDataLoaded) {
            System.err.println("Please load data first!");
            return;
        }

        try {
            // Remove instances with missing values for specified attributes
            for (int index : attributeIndices) {
                if (index >= 0 && index < data.numAttributes()) {
                    data.deleteWithMissing(index);
                } else {
                    System.err.println("Invalid attribute index: " + index);
                }
            }

            // Standardize numeric attributes
            Standardize standardize = new Standardize();
            standardize.setInputFormat(data);
            data = Filter.useFilter(data, standardize);

            System.out.println("Data cleaning completed.");
            printDataStats();
        } catch (Exception e) {
            System.err.println("Error cleaning data: " + e.getMessage());
        }
    }
    /**
     * Save the processed data to ARFF format
     * @param outputPath path where to save the ARFF file
     */
    public void saveToArff(String outputPath) {
        if (!isDataLoaded) {
            System.err.println("Please load data first!");
            return;
        }

        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(outputPath));
            saver.writeBatch();
            System.out.println("Data saved to ARFF file: " + outputPath);
        } catch (Exception e) {
            System.err.println("Error saving ARFF file: " + e.getMessage());
        }
    }

    /**
     * Print basic statistics about the dataset
     */
    public void printDataStats() {
        if (!isDataLoaded) {
            System.err.println("No data loaded!");
            return;
        }

        System.out.println("\n=== Dataset Statistics ===");
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());

        // Print attribute information
        System.out.println("\nAttributes:");
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println("- " + data.attribute(i).name());
        }
    }
}