import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.io.File;


public class ClassifierGUI extends JFrame {
    private ClassificationFramework framework;
    private JTextArea logArea;
    private JButton loadDataButton;
    private JButton trainButton;
    private JComboBox<String> classifierComboBox;
    private JLabel datasetLabel;
    private String currentDataPath;

    public ClassifierGUI() {
        framework = new ClassificationFramework();
        initializeGUI();
    }

    private void initializeGUI() {
        setTitle("WEKA Classifier Interface");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1000, 800);
        setLocationRelativeTo(null);


        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(new EmptyBorder(10, 10, 10, 10));


        JPanel controlPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 5));


        datasetLabel = new JLabel("No dataset loaded");
        loadDataButton = new JButton("Load Dataset");
        loadDataButton.addActionListener(e -> loadDataset());


        classifierComboBox = new JComboBox<>(new String[]{
                "Linear Regression",
                "SVM Regression",
                "M5P Decision Tree",
                "Random Forest"
        });


        trainButton = new JButton("Train Model");
        trainButton.addActionListener(e -> trainModel());
        trainButton.setEnabled(false);

        // Add components to control panel
        controlPanel.add(loadDataButton);
        controlPanel.add(datasetLabel);
        controlPanel.add(new JLabel("Classifier:"));
        controlPanel.add(classifierComboBox);
        controlPanel.add(trainButton);


        logArea = new JTextArea();
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 20));
        JScrollPane scrollPane = new JScrollPane(logArea);

        // Add all components to main panel
        mainPanel.add(controlPanel, BorderLayout.NORTH);
        mainPanel.add(scrollPane, BorderLayout.CENTER);

        // Status bar
        JPanel statusBar = new JPanel(new FlowLayout(FlowLayout.LEFT));
        statusBar.setBorder(BorderFactory.createLoweredBevelBorder());
        statusBar.add(new JLabel("Ready"));
        mainPanel.add(statusBar, BorderLayout.SOUTH);


        add(mainPanel);


        redirectSystemOut();
    }

    private void loadDataset() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            public boolean accept(File f) {
                return f.isDirectory() || f.getName().toLowerCase().endsWith(".arff");
            }
            public String getDescription() {
                return "ARFF Files (*.arff)";
            }
        });

        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            currentDataPath = selectedFile.getAbsolutePath();
            try {
                framework.loadData(currentDataPath);
                datasetLabel.setText("Dataset: " + selectedFile.getName());
                trainButton.setEnabled(true);
                logArea.append("Dataset loaded successfully: " + selectedFile.getName() + "\n");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this,
                        "Error loading dataset: " + e.getMessage(),
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
                logArea.append("Error loading dataset: " + e.getMessage() + "\n");
            }
        }
    }

    private void trainModel() {
        String selectedClassifier = (String) classifierComboBox.getSelectedItem();


        framework = new ClassificationFramework();

        // Add selected classifier
        try {
            switch (selectedClassifier) {
                case "Linear Regression":
                    framework.addClassifier(new LinearRegressionClassifier());
                    break;
                case "SVM Regression":
                    framework.addClassifier(new SVMRegressionClassifier());
                    break;
                case "M5P Decision Tree":
                    framework.addClassifier(new M5PTreeClassifier());
                    break;
                case "Random Forest":
                    framework.addClassifier(new RandomForestClassifier());
                    break;
            }

            // Disable UI elements during training
            trainButton.setEnabled(false);
            loadDataButton.setEnabled(false);
            classifierComboBox.setEnabled(false);

            // Run training in background thread
            new SwingWorker<Void, Void>() {
                @Override
                protected Void doInBackground() throws Exception {
                    framework.loadData(currentDataPath);
                    framework.trainAndEvaluate();
                    return null;
                }

                @Override
                protected void done() {
                    trainButton.setEnabled(true);
                    loadDataButton.setEnabled(true);
                    classifierComboBox.setEnabled(true);
                }
            }.execute();

        } catch (Exception e) {
            JOptionPane.showMessageDialog(this,
                    "Error during training: " + e.getMessage(),
                    "Error",
                    JOptionPane.ERROR_MESSAGE);
            logArea.append("Error during training: " + e.getMessage() + "\n");
        }
    }

    private void redirectSystemOut() {
        // Create a custom output stream that writes to the log area
        System.setOut(new java.io.PrintStream(new java.io.OutputStream() {
            @Override
            public void write(int b) {
                logArea.append(String.valueOf((char) b));
                logArea.setCaretPosition(logArea.getDocument().getLength());
            }
        }));
    }

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }

        SwingUtilities.invokeLater(() -> {
            ClassifierGUI gui = new ClassifierGUI();
            gui.setVisible(true);
        });
    }
}