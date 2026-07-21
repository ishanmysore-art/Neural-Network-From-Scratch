import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.io.*;

/*
 * @author Ishan Mysore
 * @date 11/24/25
 * @file NLayer.java
 * 
 * This code implements the backpropagation training algorithm for training neural networks in Java, for any number of  
 * layers N. The code separates functionality into different methods, including configuration, memory allocation, 
 * weight initialization, training, and running the network. The system supports independent runtime modes for 
 * training and running, outputs full configuration and results, and handles user-selectable weight initialization
 * (manual, randomized, or file input). Finally, this code demonstrates functionality on standard Boolean test cases
 * (AND, OR, XOR) with specified parameters.
 * 
 * This code also adds how much time it took to perform the training/running along with the hyperbolic tangent threshold function.
 * Additionally, the code implements the use of a control file to tell the program what to do at execution time.
 * Lastly, this code implements saving and loading weights to/from a binary file.
 * 
 * All features are implemented as per the project specification and design document.
 * 
 * Table of Contents:
 * 
 * public static void loadConfigParams()
 * public static void initializeRNG()
 * public static void echoConfigParams()
 * public static void allocateMemory()
 * public static void saveWeights(String filename)
 * public static void loadWeights(String filename)
 * public static void populateArrays()
 * public static void setInputTable(String filename)
 * public static void setInputTableFromFolder(String folderName)
 * public static void setTruthTable(String filename)
 * public static void setWeightsManually()
 * public static void randomizeWeights(double low, double high)
 * public static double randRange(double low, double high)
 * public static double sigmoid(double x)
 * public static double sigmoidDeriv(double x)
 * public static double tanh(double x)
 * public static double tanhDeriv(double x)
 * public static double activation(double x)
 * public static double activationDeriv(double x)
 * public static void setUpRunActivations(double[] input)
 * public static double runForTraining(int caseNum)
 * public static void runForRunning()
 * public static void setUpTrainActivations(int n)
 * public static void trainOnce()
 * public static void trainAllTestCases()
 * public static void printTrainError()
 * public static void runAllTestCases()
 * public static void printResults()
 * public static void main(String[] args)
 */
public class NLayer
{
   public static final String CONFIG_FILE_NAME_DEFAULT = "configFile.txt";
/*
* Configuration parameters (loaded from configuration file)
*/
   public static String configFileName;
   public static int numConnectivityLayers;  // number of connectivity layers in the network
   public static int numInput;               // number of input activations (a[m])
   public static int numOutput;              // number of output activations (a[OUTPUTLAYER][i])
   public static boolean modeTrain;          // true = training; false = running
   public static boolean showInputTable;     // option for run report
   public static boolean showTruthTable;     // option for run report (running does not require truth table)
   public static double randLow;             // lowest value for random generation
   public static double randHigh;            // highest value for random generation
   public static double lambda;              // learning rate
   public static int maxIters;               // maximum number of iterations that the network will train on
   public static int keepAlive;              // number of iterations between messages (or no output if it is set to zero)
   public static double errorThreshold;      // average error threshold (stop when average error <= this value)
   public static int numCases;               // for boolean problems: 4 cases 00,01,10,11 in this order
   public static boolean saveWeights;        // user-selectable option to save the weights
   public static String choosingWeights;     // LOAD, MANUAL, or RANDOM
   public static String fileNameLoad;        // file to load weights
   public static String fileNameSave;        // file to save weights
   public static String fileNameTruth;       // file to load truth table from
   public static String fileNameInput;       // file to load input table from
   public static String fileOrFolder;        // FILE or FOLDER for input table
/*
* Order of the constants in the configuration file.
* This is to avoid "magic numbers" in the loadConfigParams method.
*/
   private static final int IDX_NUM_LAYERS = 0;
   private static final int IDX_CONFIG = 1;
   private static final int IDX_MODE_TRAIN = 2;
   private static final int IDX_RAND_LOW = 3;
   private static final int IDX_RAND_HIGH = 4;
   private static final int IDX_LAMBDA = 5;
   private static final int IDX_MAX_ITERS = 6;
   private static final int IDX_KEEP_ALIVE = 7;
   private static final int IDX_ERROR_THRESHOLD = 8;
   private static final int IDX_SHOW_INPUT_TABLE = 9;
   private static final int IDX_SHOW_TRUTH_TABLE = 10;
   private static final int IDX_NUM_CASES = 11;
   private static final int IDX_CHOOSING_WEIGHTS = 12;
   private static final int IDX_SAVE_WEIGHTS = 13;
   private static final int IDX_FILE_NAME_LOAD = 14;
   private static final int IDX_FILE_NAME_SAVE = 15;
   private static final int IDX_FILE_NAME_TRUTH = 16;
   private static final int IDX_FILE_NAME_INPUT = 17;  
   private static final int IDX_FILE_OR_FOLDER = 18;
/*
 * Activation and connectivity layers
 */
   public static int numActivationLayers;
   public static int numHiddenLayers;
   public static int outputLayer;
   private static final int INPUT_LAYER = 0;
   private static final int FIRST_HIDDEN_LAYER = 1;
   private static final int NUM_INPUT_LAYERS = 1;
   private static final int NUM_OUTPUT_LAYERS = 1;
/*
* Network arrays (allocated in allocateMemory)
*/
   public static int[] config;           // configuration array for any number of layers
   public static double[][] a;           // input activations
   public static double[][][] w;         // w[n][k][j] weights, where k and j are abstractions of any two adjacent layers
   public static double[][] theta;       // two-dimensional theta array for training only
   public static double[][] psi;         // two-dimensional psi array for training only
   public static double[][] testCases;   // size [numCases][numInput]
   public static double[][] truth;       // truth table (targets) size [numCases] (training-only)
   public static double[][] outputs;     // the values that the neural network outputs after running
/*
* Output parameters
*/
   public static double avgError;         // average training error
   public static int iterations;          // number of training iterations
   public static boolean reachedError;    // tracks whether we are under the error threshold
   public static boolean reachedMaxIter;  // tracks whether we have reached the max number of iterations
   public static long startTime;          // start time of training/running
   public static long endTime;            // end time of training/running
/*
* Other miscellaneous parameters
*/
   public static Random rng;              // Random Number Generator

/*
 * Load configuration parameters from a text file.
 * Each line in the file corresponds to a specific parameter in the following order:
 * 0: numConnectivityLayers
 * 1: config (dash-separated string)
 * 2: modeTrain (true/false)
 * 3: randLow (double)
 * 4: randHigh (double)
 * 5: lambda (double)
 * 6: maxIters (int)
 * 7: keepAlive (int)
 * 8: errorThreshold (double)
 * 9: showInputTable (true/false)
 * 10: showTruthTable (true/false)
 * 11: numCases (int)
 * 12: choosingWeights (LOAD/MANUAL/RANDOM)
 * 13: saveWeights (true/false)
 * 14: fileNameLoad (string)
 * 15: fileNameSave (string)
 * 16: fileNameTruth (string)
 * 17: fileNameInput (string)
 * @postcondition all configuration parameters have been set from the file.
 */
   public static void loadConfigParams()
   {
      try (BufferedReader br = new BufferedReader(new FileReader(configFileName)))
      {
         String line;
         int lineCount = 0;

         while ((line = br.readLine()) != null)
         {
            line = line.trim();                          // trim whitespace
            int semicolonIndex = line.indexOf(';');   // find the index of the semicolon
            String value = line.substring(0, semicolonIndex).trim();

            switch (lineCount)
            {
               case IDX_NUM_LAYERS:
                  numConnectivityLayers = Integer.parseInt(value);
                  numActivationLayers = numConnectivityLayers + NUM_INPUT_LAYERS;
                  numHiddenLayers = numConnectivityLayers - NUM_OUTPUT_LAYERS;
                  outputLayer = numActivationLayers - 1;
                  break;

               case IDX_CONFIG:
                  String[] parts = value.split("-");                 // IMPORTANT: ASK IF WE CAN CREATE ARRAYS IN PARSER

                  if (parts.length != numActivationLayers) // safety check
                  {
                     throw new IllegalArgumentException(
                        "Config error: expected " + numActivationLayers + " values but found " + parts.length);
                  }

                  config = new int[numActivationLayers];
                  for (int n = INPUT_LAYER; n < numActivationLayers; n++) config[n] = Integer.parseInt(parts[n]);

                  numInput = config[INPUT_LAYER];
                  numOutput = config[outputLayer];
                  break;

               case IDX_MODE_TRAIN:  modeTrain = Boolean.parseBoolean(value); break;
               case IDX_RAND_LOW:  randLow = Double.parseDouble(value); break;
               case IDX_RAND_HIGH:  randHigh = Double.parseDouble(value); break;
               case IDX_LAMBDA:  lambda = Double.parseDouble(value); break;
               case IDX_MAX_ITERS:  maxIters = Integer.parseInt(value); break;
               case IDX_KEEP_ALIVE:  keepAlive = Integer.parseInt(value); break;
               case IDX_ERROR_THRESHOLD:  errorThreshold = Double.parseDouble(value); break;
               case IDX_SHOW_INPUT_TABLE:  showInputTable = Boolean.parseBoolean(value); break;
               case IDX_SHOW_TRUTH_TABLE: showTruthTable = Boolean.parseBoolean(value); break;
               case IDX_NUM_CASES: numCases = Integer.parseInt(value); break;
               case IDX_CHOOSING_WEIGHTS: choosingWeights = value; break;
               case IDX_SAVE_WEIGHTS: saveWeights = Boolean.parseBoolean(value); break;
               case IDX_FILE_NAME_LOAD: fileNameLoad = value; break;
               case IDX_FILE_NAME_SAVE: fileNameSave = value; break;
               case IDX_FILE_NAME_TRUTH: fileNameTruth = value; break;
               case IDX_FILE_NAME_INPUT: fileNameInput = value; break;
               case IDX_FILE_OR_FOLDER: fileOrFolder = value; break;
               default: break;
            } // switch (lineCount)

            lineCount++;
         } // while ((line = br.readLine()) != null)
      } // try (BufferedReader br = new BufferedReader(new FileReader(configFileName)))
      catch (IOException e)
      {
         e.printStackTrace();
      }
   } // public static void loadConfigParams()

/*
* Initialize the random number generator.
* @postcondition the random number generator has been initialized.
*/
   public static void initializeRNG()
   {
      rng = new Random();
   }

/*
* Print the configuration parameters in a readable format for the user.
*/
   public static void echoConfigParams()
   {
      System.out.println("Time Started: " + LocalDateTime.now());   // print the current date and time
      System.out.println("=======================================================================================================");
      System.out.println("--- Configuration Parameters ---");
      System.out.println("Configuration File Name: " + configFileName);
      System.out.println("Number of Connectivity Layers: " + numConnectivityLayers);
      System.out.print("Network Configuration: ");

      for (int n = INPUT_LAYER; n < numActivationLayers; n++)
      {
         System.out.print(config[n]);
         if (n < numConnectivityLayers) System.out.print("-");
      }

      System.out.println("\nMode: " + (modeTrain ? "TRAIN" : "RUN"));
      System.out.println("Populate Method: " + choosingWeights);

      if (choosingWeights.equals("RANDOM"))
      {
         System.out.println(String.format("Random range: [%.4f, %.4f]", randLow, randHigh));
      }

      System.out.println("Show Input Table: " + showInputTable);
      System.out.println("Show Truth Table: " + showTruthTable);

      if (modeTrain)
      {
         System.out.println(String.format("Learning Rate: %.6f", lambda));
         System.out.println("Max Iterations: " + maxIters);
         System.out.println("Keep Alive Interval: " + keepAlive);
         System.out.println(String.format("Average error threshold: %.8f", errorThreshold));
         System.out.println("Truth Table File: " + fileNameTruth);
      }

      System.out.println("Input Table Folder: " + fileNameInput);
      System.out.println("Number of Cases: " + numCases);
      System.out.println("Save Weights: " + saveWeights);

      if (choosingWeights.equals("LOAD"))
      {
         System.out.println("Loading Weights From File: " + fileNameLoad);
      }

      if (saveWeights)
      {
         System.out.println("Saving Weights To File: " + fileNameSave);
      }
   } // public static void echoConfigParams()

/*
* Allocate memory for each of the network arrays listed above.
* @postcondition all of the arrays have memory allocated
*/
   public static void allocateMemory()
   {
      a = new double[numActivationLayers][];
      for (int n = INPUT_LAYER; n < numActivationLayers; n++) a[n] = new double[config[n]];
      
      w = new double[numActivationLayers][][];
      for (int n = FIRST_HIDDEN_LAYER; n < numActivationLayers; n++) w[n] = new double[config[n-1]][config[n]];
      
      outputs = new double[numCases][numOutput];
      testCases = new double[numCases][numInput];

      if (modeTrain)                           // Training-only arrays: allocate only when training
      {
         theta = new double[numConnectivityLayers][];
         for (int n = FIRST_HIDDEN_LAYER; n < numConnectivityLayers; n++) theta[n] = new double[config[n]];

         psi = new double[numActivationLayers][];
         for (int n = FIRST_HIDDEN_LAYER; n < numActivationLayers; n++) psi[n] = new double[config[n]];

         truth = new double[numCases][numOutput];
      } // if (modeTrain)
   } // public static void allocateMemory()

/*
 * Save the current network weights to a binary file.
 * The file format includes the network configuration (numInput, numHidden, numOutput)
 * followed by all weights in a consistent order:
 *   - w[HIDDENLAYERJ][k][j] (hidden × input)
 *   - w[OUTPUTLAYER][j][i] (hidden → output)
 *
 * @param filename the binary file to save to
 */
public static void saveWeights(String filename)
{
   try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename)))
   {
      for (int n = INPUT_LAYER; n < numActivationLayers; n++)
      {
         dos.writeInt(config[n]);
      }

      for (int n = FIRST_HIDDEN_LAYER; n < numActivationLayers; n++)
      {
         for (int k = 0; k < config[n-1]; k++)
         {
            for (int j = 0; j < config[n]; j++)
            {
               dos.writeDouble(w[n][k][j]);
            }
         }
      }

      System.out.println("Weights saved successfully to " + filename);
   } // try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename)))
   catch (IOException e)
   {
      System.err.println("Error saving weights: " + e.getMessage());
   }
} // public static void saveWeights(String filename)

/*
 * Load network weights and biases from a binary file.
 * The file format must match what was written by saveWeights().
 * Configuration is validated against the current network.
 *
 * @param filename the binary file to load from
 */
public static void loadWeights(String filename)
{
   try (DataInputStream dis = new DataInputStream(new FileInputStream(filename)))
   {
      for (int n = INPUT_LAYER; n < numActivationLayers; n++)
      {
         int value = dis.readInt();
         if (value != config[n])
         {
            System.out.println(value);
            System.out.println(config[n]);
            throw new IllegalStateException("Weight file config does not match the expected config.");
         }
      }

      for (int n = FIRST_HIDDEN_LAYER; n < numActivationLayers; n++)
      {
         for (int k = 0; k < config[n-1]; k++)
         {
            for (int j = 0; j < config[n]; j++)
            {
               w[n][k][j] = dis.readDouble();
            }
         }
      }

      System.out.println("Weights loaded successfully from " + filename);
   } // try (DataInputStream dis = new DataInputStream(new FileInputStream(filename)))
   catch (IOException e)
   {
      System.err.println("Error loading weights: " + e.getMessage());
   }
} // public static void loadWeights(String filename)

/*
* Populate values for each of the arrays for which memory was allocated earlier.
* Also set the weights manually/randomize them, depending on the corresponding parameters.
* If both preloadWeights and populateRandom are false, set the weights to zero.
*/
   public static void populateArrays()
   {
      if (fileOrFolder.equals("FILE")) setInputTable(fileNameInput);
      else if (fileOrFolder.equals("FOLDER")) setInputTableFromFolder(fileNameInput);

      if (modeTrain) setTruthTable(fileNameTruth);

/*
* Input the weights according to value of user-selectable option.
*/
      if (choosingWeights.equals("LOAD"))
      {
         loadWeights(fileNameLoad);
      }
      else if (choosingWeights.equals("MANUAL"))
      {
         setWeightsManually();
      }
      else
      {
         randomizeWeights(randLow, randHigh);
      }
   } // public static void populateArrays()

/*
 * Load the input table from an input text file.
 */
   public static void setInputTable(String filename)
   {
      try (BufferedReader br = new BufferedReader(new FileReader(filename)))
      {
         String line;
         int row = 0;
         
         while ((line = br.readLine()) != null && row < testCases.length)
         {
            line = line.trim();
            String[] parts = line.split("\\s+"); // split by spaces/tabs
            
            for (int col = 0; col < parts.length && col < testCases[row].length; col++)
            {
               testCases[row][col] = Double.parseDouble(parts[col]);
            }
            
            row++;
         } // while ((line = br.readLine()) != null && row < testCases.length)
      } // try (BufferedReader br = new BufferedReader(new FileReader(filename)))
      catch (IOException e)
      {
         System.err.println("Error reading input table file: " + e.getMessage());
      }
   } // public static void setInputTable(String filename)

   public static void setInputTableFromFolder(String folderName)
   {
      File folder = new File(folderName);
      File[] files = folder.listFiles((dir, name) -> name.endsWith(".bin"));

      if (files == null || files.length == 0) {
         System.err.println("No activation files found in folder: " + folderName);
         return;
      }

      Arrays.sort(files, Comparator.comparing(File::getName));

      if (files.length != testCases.length) {
         System.err.println("Mismatch: expected " + testCases.length +
                              " files but found " + files.length);
         return;
      }

      try {
         for (int row = 0; row < files.length; row++) {
               try (DataInputStream dis =
                        new DataInputStream(new BufferedInputStream(
                           new FileInputStream(files[row])))) {

                  for (int col = 0; col < testCases[row].length; col++) {
                     testCases[row][col] = dis.readDouble();
                  }
               }
         }
         System.out.println("Input table loaded successfully from folder: " + folderName);
      }
      catch (IOException e) {
         System.err.println("Error reading activation files: " + e.getMessage());
      }
   }

/*
 * Load the truth table from a text file.
 */
   public static void setTruthTable(String filename)
   {
      try (BufferedReader br = new BufferedReader(new FileReader(filename)))
      {
         String line;
         int row = 0;

         while ((line = br.readLine()) != null && row < truth.length)
         {
            line = line.trim();
            String[] parts = line.split("\\s+"); // split on spaces or tabs

            for (int col = 0; col < parts.length && col < truth[row].length; col++)
            {
               truth[row][col] = Double.parseDouble(parts[col]);
            }

            row++;
         } // while ((line = br.readLine()) != null && row < truth.length)
      } // try (BufferedReader br = new BufferedReader(new FileReader(filename)))
      catch (IOException e)
      {
         System.err.println("Error reading truth table file: " + e.getMessage());
      }
   } // public static void setTruthTable(String filename)

/*
* Manual weight initializer that works for any N-Layer network.
*/
   public static void setWeightsManually()
   {
      for (int n = FIRST_HIDDEN_LAYER; n < numActivationLayers; n++)
      {
         for (int k = 0; k < config[n-1]; k++)
         {
            for (int j = 0; j < config[n]; j++)
            {
               w[n][k][j] = 0.1 * (j + 1) + 0.05 * (k + 1);
            }
         }
      } // for (int n = FIRST_HIDDEN_LAYER; n < numActivationLayers; n++)
   } // public static void setWeightsManually()

/*
* Select random weights for the network, with each weight ranging from the low to high values passed in as parameters.
* Calls the randRange method below to select a random value for each weight in the arrays.
* @param low the lowest possible value for the weights
* @param high the highest possible value for the weights
*/
   public static void randomizeWeights(double low, double high)
   {
      for (int n = FIRST_HIDDEN_LAYER; n < numActivationLayers; n++)
      {
         for (int k = 0; k < config[n-1]; k++)
         {
            for (int j = 0; j < config[n]; j++)
            {
               w[n][k][j] = randRange(low, high);
            }
         }
      }
   } // public static void randomizeWeights(double low, double high)

/*
* Select a random number from low to high.
* @param low the lowest possible value we want to select
* @param high the highest possible value we want to select
* @return the random value generated
*/
   public static double randRange(double low, double high)
   {
      return low + rng.nextDouble() * (high - low);
   }

/*
* Calculates the activation function (sigmoid) for an input x.
* @param x the input that we want to calculate the sigmoid of.
* @return the sigmoid function for x
*/
   public static double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/*
* Calculates the derivative of the sigmoid function for an input x.
* @param x the input that we want to calculate the sigmoid of.
* @return the derivative of the sigmoid function for x
*/
   public static double sigmoidDeriv(double x)
   {
      double sig = sigmoid(x);
      return sig * (1.0 - sig);
   }

/*
 * Calculates the hyperbolic tangent activation function for an input x.
 * @param x the input that we want to calculate the hyperbolic tangent of.
 * @return the hyperbolic tangent function for x
 */
   public static double tanh(double x)
   {
      double eps = (x < 0.0) ? 1.0 : -1.0;
      double exp2x = Math.exp(eps * 2.0 * x);
      return eps * (exp2x - 1.0) / (exp2x + 1.0);
   }

/*
 * Calculates the derivative of the hyperbolic tangent function for an input x.
 * @param x the input that we want to calculate the derivative of hyperbolic tangent of.
 * @return the derivative of the hyperbolic tangent function for x
 */
   public static double tanhDeriv(double x)
   {
      double tanh = tanh(x);
      return 1.0 - tanh * tanh;
   }

/*
 * Calculates the activation function based on user selection.
 * @param x the input value
 * @return the activated output
 */
   public static double activation(double x)
   {
      return sigmoid(x);
   }

/*
 * Calculates the derivative of the activation function based on user selection.
 * @param x the input value
 * @return  the derivative of the activated output
 */
   public static double activationDeriv(double x)
   {
      return sigmoidDeriv(x);
   }

/*
 * Set up the input activations for running the network.
 * @param input the input array to set the activations from
 */
   public static void setUpRunActivations(double[] input)
   {
      for (int x = 0; x < numInput; x++)
      {
         a[INPUT_LAYER][x] = input[x]; // copy inputs into a[x]
      }
   }

/*
* Run for training. This method only runs the network once.
* It does not depend on truth table and does no training allocations.
* theta is not local (it is an array).
* Coded directly to the design document.
*/
   public static double runForTraining(int caseNum)
   {
      for (int n = FIRST_HIDDEN_LAYER; n < outputLayer; n++)
      {
         for (int k = 0; k < config[n]; k++)
         {
            double sum = 0.0;

            for (int j = 0; j < config[n-1]; j++)
            {
               sum += a[n-1][j] * w[n][j][k];
            }

            theta[n][k] = sum;
            a[n][k] = activation(theta[n][k]);
         } // for (int k = 0; k < config[n]; k++)
      } // for (int n = INPUT_LAYER; n < numConnectivityLayers; n++)
/*
* Calculate output layer psi and return the error.
*/
      double sumError = 0.0;
      int n = outputLayer;

      for (int i = 0; i < numOutput; i++)
      {
         double theta_i = 0.0;

         for (int j = 0; j < config[n-1]; j++)
         {
            theta_i += a[n-1][j] * w[n][j][i];
         }

         a[n][i] = activation(theta_i);
         double omega_i = truth[caseNum][i] - a[n][i];
         psi[n][i] = omega_i * activationDeriv(theta_i);
         sumError += omega_i * omega_i;
      } // for (int i = 0; i < numOutput; i++)
      
      sumError /= 2.0;
      return sumError;
   } // public static double runForTraining(int caseNum)

/*
* Run for running. This method only runs the network once.
* It does not depend on truth table and does no training allocations.
* Additionally, theta_j is local.
* Coded directly to the design document.
*/
   public static void runForRunning()
   {
      for (int n = FIRST_HIDDEN_LAYER; n < numActivationLayers; n++)
      {
         for (int k = 0; k < config[n]; k++)
         {
            double theta = 0.0;

            for (int j = 0; j < config[n-1]; j++)
            {
               theta += a[n-1][j] * w[n][j][k];
            }

            a[n][k] = activation(theta);
         } // for (int k = 0; k < config[n]; k++)
      } // for (int n = FIRST_HIDDEN_LAYER; n < numConnectivityLayers; n++)
   } // public static void runForRunning()

/*
 * Set up the input activations for training the network.
 * @param n the index of the test case to set the activations from
 * @postcondition a[k] has been set to the input activations for test case n.
 */
   public static void setUpTrainActivations(int n)
   {
      for (int x = 0; x < numInput; x++)
      {
         a[INPUT_LAYER][x] = testCases[n][x];
      }
   } // public static void setUpTrainActivations(int n)

/*
* Train (gradient descent). Training is implemented exactly as the algorithm
* in the design document.
*
* @precondition in training mode; truth and F are populated.
*/
   public static void trainOnce()
   {
      for (int n = numConnectivityLayers - 1; n > FIRST_HIDDEN_LAYER; n--)
      {
         for (int k = 0; k < config[n]; k++)
         {
            double omega = 0.0;

            for (int j = 0; j < config[n+1]; j++)
            {
               omega += psi[n+1][j] * w[n+1][k][j];
               w[n+1][k][j] += lambda * a[n][k] * psi[n+1][j]; // update weights
            }

            psi[n][k] = omega * activationDeriv(theta[n][k]);
         } // for (int k = 0; k < config[n]; k++)
      } // for (int n = numConnectivityLayers - 1; n > FIRST_HIDDEN_LAYER; n--)

      int n = FIRST_HIDDEN_LAYER;

      for (int m = 0; m < config[n]; m++)
      {
         double omega = 0.0;

         for (int k = 0; k < config[n+1]; k++)
         {
            omega += psi[n+1][k] * w[n+1][m][k];
            w[n+1][m][k] += lambda * a[n][m] * psi[n+1][k]; // update weights
         }

         for (int x = 0; x < config[n-1]; x++)
         {
            w[n][x][m] += lambda * a[n-1][x] * omega * activationDeriv(theta[n][m]); // update weights
         }
      } // for (int m = 0; m < config[n]; m++)
   } // public static void trainOnce()

/*
 * Train all test cases until stopping criteria are met.
 */
   public static void trainAllTestCases()
   {
      while (!reachedError && !reachedMaxIter)     // keep going until under error threshold or until max iterations
      {
         iterations++;
         double sumError = 0.0;

         for (int t = 0; t < numCases; t++)        // loop over test cases
         {
            setUpTrainActivations(t);              // set up training
            sumError += runForTraining(t);         // Run a forward pass of the network
            trainOnce();
         } // for (int t = 0; t < numCases; t++)

         avgError = sumError / (double) numCases;
/*
* check stopping criteria
*/
         if (avgError <= errorThreshold) reachedError = true;
         if (iterations >= maxIters) reachedMaxIter = true;

         if ((keepAlive != 0) && ((iterations % keepAlive) == 0))
         {
            System.out.printf("Iteration %d, Error = %f\n", iterations, avgError);
         }
      } // while (!reachedError && !reachedMaxIter)
   } // public static void trainAllTestCases()

/*
* When in mode train, prints the output results of training (error, number of iterations, reasons for stopping).
*/
   public static void printTrainError()
   {
      if (modeTrain)
      {
/*
* Report training results (reporting not part of train() itself)
*/
         System.out.println("--- Training Exit Information ---");
         if (reachedError) System.out.println("Reason: Error threshold reached (avg error <= " + errorThreshold + ")");
         if (reachedMaxIter) System.out.println("Reason: Maximum iterations reached (" + maxIters + ")");
         System.out.println("Iterations: " + iterations);
         System.out.println(String.format("Average error: %.3E", avgError));
         System.out.println("----------------------------------");
      } // if (modeTrain)
   } // public static void printTrainError()

/*
* Run all test cases and produce reports.
*/
   public static void runAllTestCases()
   {
/*
* Loop over the number of test cases and run the network for each test case.
*/
      for (int t = 0; t < numCases; t++)
      {
         setUpRunActivations(testCases[t]);
         runForRunning(); // run the network

         for (int i = 0; i < numOutput; i++)
         {
            outputs[t][i] = a[outputLayer][i];
         }
      } // for (int t = 0; t < numCases; t++)
   } // public static void runAllTestCases()

/*
* When in mode run, prints the output results (input table, truth table, output values upon user's request)
*/
   public static void printResults()
   {
      System.out.println("--- Run Results ---");
      System.out.println("Time taken (ms): " + (endTime - startTime));
/*
* Print each value in the input table (if the user wants to show the input table)
*/
      if (showInputTable)
      {
         System.out.println("Input Table:"); // this prints the inputs in order
         for (int t = 0; t < numCases; t++)
         {
            for (int x = 0; x < numInput; x++) System.out.print((int) testCases[t][x] + " ");
            System.out.println();
         }
      } // if (showInputTable)
/*
* Print each value in the truth table (if the user want to show the truth table)
*/
      if (showTruthTable && truth != null)
      {
         System.out.println("Truth Table:");
         for (int t = 0; t < numCases; t++)
         {
            System.out.print(String.format("Input: %d %d ->", 
            (int) testCases[t][0], (int) testCases[t][1]));
      
            for (int i = 0; i < numOutput; i++)
            {
               System.out.print(" " + (int) truth[t][i]);
            }

            System.out.println();
         } // for (int t = 0; t < numCases; t++)
      } // if (showTruthTable && truth != null)
/*
* Prints the output values.
*/
      System.out.println("Actual Outputs:");
      for (int t = 0; t < numCases; t++)
      {

         for (int i = 0; i < numOutput; i++)
         {
            System.out.print(String.format("%.2f ", outputs[t][i]));
         }

         System.out.println();
      } // for (int t = 0; t < numCases; t++)
      System.out.println("--------------------");
   } // public static void printResults()

/*
* Puts it all together!
*/
   public static void main(String[] args)
   {
      if (args.length == 0)
      {
         configFileName = CONFIG_FILE_NAME_DEFAULT;
      }
      else
      {
         configFileName = args[0];
      }
      
      loadConfigParams();
      echoConfigParams();
      initializeRNG();
      allocateMemory();
      populateArrays();
      startTime = System.currentTimeMillis();

      if (modeTrain)
      {
         trainAllTestCases();
         printTrainError();
         if (saveWeights) saveWeights(fileNameSave);
      }

      runAllTestCases();
      endTime = System.currentTimeMillis();
      printResults();
   } // public static void main(String[] args)
} // public class NLayer