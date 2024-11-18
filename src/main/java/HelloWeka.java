import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

public class HelloWeka {
    public static void main(String[] args) {
        try {
            // Load ARFF file
            String dataPath = "data/iris.arff";
            DataSource source = new DataSource(dataPath);
            Instances data = source.getDataSet();

            // Set the class index (last attribute as target)
            data.setClassIndex(data.numAttributes() - 1);

            // Option 1: Evaluate with a training/testing split (80%/20%)
            System.out.println("Evaluating with Training/Testing Split (80/20)...");
            evaluateWithSplit(data);

            // Option 2: Evaluate with 10-fold cross-validation
            System.out.println("\nEvaluating with 10-Fold Cross-Validation...");
            evaluateWithCrossValidation(data);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Evaluate classifiers using a training/testing split (80% training, 20% testing)
     *
     * @param data the dataset
     */
    private static void evaluateWithSplit(Instances data) {
        try {
            // Split data into training (80%) and testing (20%)
            data.randomize(new java.util.Random(1));
            int trainSize = (int) Math.round(data.numInstances() * 0.8);
            int testSize = data.numInstances() - trainSize;

            Instances trainingData = new Instances(data, 0, trainSize);
            Instances testingData = new Instances(data, trainSize, testSize);

            // Train and test OneR Classifier
            OneRClassifier oneR = new OneRClassifier();
            System.out.println("Training OneR Classifier...");
            oneR.buildClassifier(trainingData);
            System.out.println("Evaluating OneR Classifier...");
            evaluateModel(oneR, testingData);

            // Train and test NaiveBayes Classifier
            NaiveBayesClassifier naiveBayes = new NaiveBayesClassifier();
            System.out.println("Training NaiveBayes Classifier...");
            naiveBayes.buildClassifier(trainingData);
            System.out.println("Evaluating NaiveBayes Classifier...");
            evaluateModel(naiveBayes, testingData);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Evaluate classifiers using 10-fold cross-validation
     *
     * @param data the dataset
     */
    private static void evaluateWithCrossValidation(Instances data) {
        try {
            // Initialize classifiers
            OneRClassifier oneR = new OneRClassifier();
            NaiveBayesClassifier naiveBayes = new NaiveBayesClassifier();

            // Perform 10-fold cross-validation for OneR Classifier
            System.out.println("Evaluating OneR Classifier...");
            evaluateWithCrossValidationHelper(oneR, data);

            // Perform 10-fold cross-validation for NaiveBayes Classifier
            System.out.println("Evaluating NaiveBayes Classifier...");
            evaluateWithCrossValidationHelper(naiveBayes, data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    private static void evaluateWithCrossValidationHelper(IClassifier classifierWrapper, Instances data) {
        try {
            // Create a new Evaluation object for the dataset
            Evaluation eval = new Evaluation(data);

            // Perform 10-fold cross-validation
            int numFolds = 10;
            data.randomize(new java.util.Random(42));
            for (int i = 0; i < numFolds; i++) {
                // Create training and testing sets for the fold
                Instances trainingData = data.trainCV(numFolds, i);
                Instances testingData = data.testCV(numFolds, i);

                // Build the classifier for this fold
                classifierWrapper.buildClassifier(trainingData);

                // Evaluate on the test set
                eval.evaluateModel(classifierWrapper.getClassifier(), testingData);
            }

            // Print evaluation results
            System.out.println(eval.toSummaryString("\n=== Cross-Validation Summary ===\n", false));
            System.out.println("Confusion Matrix:");
            double[][] confusionMatrix = eval.confusionMatrix();
            for (double[] row : confusionMatrix) {
                for (double value : row) {
                    System.out.print((int) value + "\t");
                }
                System.out.println();
            }
        } catch (Exception e) {
            System.err.println("Error evaluating classifier: " + e.getMessage());
        }
    }


    /**
     * Evaluate a classifier on testing data
     *
     * @param classifier the classifier to evaluate
     * @param data       the testing dataset
     */
    private static void evaluateModel(IClassifier classifier, Instances data) {
        try {
            int correct = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                double[] prediction = classifier.predictInstance(data.instance(i));
                int predictedClass = indexOfMax(prediction);
                int actualClass = (int) data.instance(i).classValue();

                if (predictedClass == actualClass) {
                    correct++;
                }
            }
            double accuracy = (double) correct / data.numInstances() * 100.0;
            System.out.println("Accuracy: " + accuracy + "%");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Find the index of the maximum value in an array
     *
     * @param array the array of values
     * @return the index of the maximum value
     */
    private static int indexOfMax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
