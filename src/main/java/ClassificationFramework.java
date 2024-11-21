import weka.classifiers.Evaluation;
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

    public ClassificationFramework() {
        classifiers = new ArrayList<>();
    }

    public void addClassifier(IClassifier classifier) {
        classifiers.add(classifier);
    }

    public void loadData(String dataPath) throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(dataPath));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Normalize/standardize data
        Standardize standardize = new Standardize();
        standardize.setInputFormat(data);
        data = Filter.useFilter(data, standardize);

        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;

        trainingData = new Instances(data, 0, trainSize);
        testingData = new Instances(data, trainSize, testSize);
    }

    public void trainAndEvaluate() throws Exception {
        for (IClassifier classifier : classifiers) {
            // Train
            classifier.buildClassifier(trainingData);

            // Evaluate
            Evaluation eval = new Evaluation(trainingData);
            eval.evaluateModel(classifier.getClassifier(), testingData);

            // Print results
            System.out.println("\n=== " + classifier.getModeName() + " Results ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
        }
    }

    public Instances getTrainingData() {
        return trainingData;
    }

    public Instances getTestingData() {
        return testingData;
    }
}