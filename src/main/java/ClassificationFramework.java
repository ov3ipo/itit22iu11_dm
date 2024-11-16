import weka.core.Instances;
import weka.core.converters.ArffLoader;

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
        // Load data
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(dataPath));
        Instances data = loader.getDataSet();

        // Set class index to last attribute
        data.setClassIndex(data.numAttributes() - 1);

        // Split into training and testing sets (80-20 split)
        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;

        trainingData = new Instances(data, 0, trainSize);
        testingData = new Instances(data, trainSize, testSize);
    }
}
