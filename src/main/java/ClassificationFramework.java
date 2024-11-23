import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ClassificationFramework {
    private List<IClassifier> classifiers;
    private Instances trainingData;
    private Instances testingData;
    private Instances fullData;

    public ClassificationFramework() {
        classifiers = new ArrayList<>();
    }

    public void addClassifier(IClassifier classifier) {
        classifiers.add(classifier);
    }

    public void loadData(String dataPath) throws Exception {
        System.out.println("Loading data from: " + dataPath);

        // Load data
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(dataPath));
        fullData = loader.getDataSet();
        fullData.setClassIndex(fullData.numAttributes() - 1);

        // Normalize/standardize data
        Standardize standardize = new Standardize();
        standardize.setInputFormat(fullData);
        fullData = Filter.useFilter(fullData, standardize);

        // Split into training and testing sets
        fullData.randomize(new Random(42));
        int trainSize = (int) Math.round(fullData.numInstances() * 0.8);
        int testSize = fullData.numInstances() - trainSize;

        trainingData = new Instances(fullData, 0, trainSize);
        testingData = new Instances(fullData, trainSize, testSize);

        System.out.println("Data loaded successfully.");
        System.out.println("Total instances: " + fullData.numInstances());
        System.out.println("Training instances: " + trainingData.numInstances());
        System.out.println("Testing instances: " + testingData.numInstances());
    }

    public void trainAndEvaluate() throws Exception {
        if (classifiers.isEmpty()) {
            throw new IllegalStateException("No classifiers added. Please add classifiers first.");
        }

        if (fullData == null) {
            throw new IllegalStateException("No data loaded. Please load data first.");
        }

        for (IClassifier classifier : classifiers) {
            System.out.println("\n=== Starting " + classifier.getModeName() + " Evaluation ===\n");

            // Build classifier first
            long startTime = System.currentTimeMillis();
            classifier.buildClassifier(trainingData);
            long buildTime = System.currentTimeMillis() - startTime;

            // Print run information
            System.out.println("=== Run Information ===");
            System.out.println("Scheme:       " + classifier.getClassifier().getClass().getName());
            System.out.println("Relation:     " + fullData.relationName());
            System.out.println("Instances:    " + fullData.numInstances());
            System.out.println("Attributes:   " + fullData.numAttributes());

            for (int i = 0; i < fullData.numAttributes(); i++) {
                System.out.println("              " + fullData.attribute(i).name());
            }

            System.out.printf("Time taken to build model: %.2f seconds\n", buildTime/1000.0);
            System.out.println("\nTest mode:    10-fold cross-validation");


            // Cross-validation evaluation
            System.out.println("\n=== Cross-validation ===");
            Evaluation crossValidation = new Evaluation(fullData);
            crossValidation.crossValidateModel(classifier.getClassifier(), fullData, 10, new Random(1));

            System.out.println("=== Summary ===");
            printEvaluationResults(crossValidation, fullData.numInstances());
        }
    }

    private void printEvaluationResults(Evaluation eval, int numInstances) throws Exception {
        System.out.printf("Correlation coefficient                  %.4f\n", eval.correlationCoefficient());
        System.out.printf("Mean absolute error                      %.4f\n", eval.meanAbsoluteError());
        System.out.printf("Root mean squared error                  %.4f\n", eval.rootMeanSquaredError());
        System.out.printf("Relative absolute error                 %.4f %%\n", eval.relativeAbsoluteError());
        System.out.printf("Root relative squared error             %.4f %%\n", eval.rootRelativeSquaredError());
        System.out.printf("Total Number of Instances             %d\n", numInstances);
    }

    public Instances getTrainingData() {
        return trainingData;
    }

    public Instances getTestingData() {
        return testingData;
    }

    public Instances getFullData() {
        return fullData;
    }
}