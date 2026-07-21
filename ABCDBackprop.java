import java.time.LocalDateTime;
import java.util.Random;
import java.io.*;

/*
 * @author Ishan Mysore
 * @date 10/30/25
 * @file ABCDBackprop.java
 * 
 * This code implements the backpropagation training algorithm for training neural networks in Java, restricted to an 
 * A-B-C-D architecture (A inputs, B nodes in hidden layer 1, C nodes in hidden layer 2, D outputs). The code separates 
 * functionality into different methods, including configuration, memory allocation, weight initialization, training, 
 * and running the network. The system supports independent runtime modes for training and running, outputs full
 * configuration and results, and handles user-selectable weight initialization (manual, randomized, or file input).
 * Finally, this code demonstrates functionality on standard Boolean test cases (AND, OR, XOR) with specified parameters.
 * 
 * This code also adds how much time it took to perform the training/running along with the hyperbolic tangent threshold function.
 * Additionally, the code implements the use of a control file to tell the program what to do at execution time.
 * Lastly, this code implements saving and loading weights to/from a binary file.
 * 
 * All features are implemented as per the project specification and design document.
 */
public class ABCDBackprop
{
   public static final String CONFIG_FILE_NAME_DEFAULT = "configFile.txt";
   public static String configFileName;
/*
* Configuration parameters (loaded from configuration file)
*/
   public static int numLayers;           // number of layers in the network (excluding input layer)
   public static int numInput;            // number of input activations (a[m])
   public static int numHiddenK;          // number of hidden activations (h[k]) in the first hidden layer
   public static int numHiddenJ;          // number of hidden activations (h[j]) in the second hidden layer
   public static int numOutput;           // number of output activations (a[OUTPUTLAYER][i])
   public static boolean modeTrain;       // true = training; false = running
   public static boolean showInputTable;  // option for run report
   public static boolean showTruthTable;  // option for run report (running does not require truth table)
   public static double randLow;          // lowest value for random generation
   public static double randHigh;         // highest value for random generation
   public static double lambda;           // learning rate
   public static int maxIters;            // maximum number of iterations that the network will train on
   public static double errorThreshold;   // average error threshold (stop when average error <= this value)
   public static int numCases;            // for boolean problems: 4 cases 00,01,10,11 in this order
   public static boolean saveWeights;     // user-selectable option to save the weights
   public static String choosingWeights;  // LOAD, MANUAL, or RANDOM
   public static String fileNameLoad;     // file to load weights
   public static String fileNameSave;     // file to save weights
   public static String fileNameTruth;    // file to load truth table from
   public static String fileNameInput;    // file to load input table from
/*
* Order of the constants in the configuration file.
* This is to avoid "magic numbers" in the loadConfigParams method.
*/
   private static final int IDX_NUM_LAYERS = 0;
   private static final int IDX_NUM_INPUT = 1;
   private static final int IDX_NUM_HIDDENK = 2;
   private static final int IDX_NUM_HIDDENJ = 3;
   private static final int IDX_NUM_OUTPUT = 4;
   private static final int IDX_MODE_TRAIN = 5;
   private static final int IDX_RAND_LOW = 6;
   private static final int IDX_RAND_HIGH = 7;
   private static final int IDX_LAMBDA = 8;
   private static final int IDX_MAX_ITERS = 9;
   private static final int IDX_ERROR_THRESHOLD = 10;
   private static final int IDX_SHOW_INPUT_TABLE = 11;
   private static final int IDX_SHOW_TRUTH_TABLE = 12;
   private static final int IDX_NUM_CASES = 13;
   private static final int IDX_CHOOSING_WEIGHTS = 14;
   private static final int IDX_SAVE_WEIGHTS = 15;
   private static final int IDX_FILE_NAME_LOAD = 16;
   private static final int IDX_FILE_NAME_SAVE = 17;
   private static final int IDX_FILE_NAME_TRUTH = 18;
   private static final int IDX_FILE_NAME_INPUT = 19;
/*
* Constants corresponding to the layers in the network.
*/
   private static final int INPUTLAYER = 0;
   private static final int HIDDENLAYERK = 1;
   private static final int HIDDENLAYERJ = 2;
   private static final int OUTPUTLAYER = 3;
/*
 * Distinction between input and connectivity layers.
 */
   public static int numConnectivityLayers;
   private static final int NUMINPUTLAYERS = 1;
/*
* Network arrays (allocated in allocateMemory)
*/
   public static double[][] a;           // activations
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
* Set hard-coded values for each of the configuration parameters.
* @postcondition all the configuration parameters have been set to user values.
*/
   public static void setConfigParams()
   {
      numLayers = 4;
      numInput = 2;
      numHiddenK = 3;
      numHiddenJ = 3;
      numOutput = 3;
      modeTrain = true;
      randLow = 0.1;
      randHigh = 1.5;
      lambda = 0.3;
      maxIters = 100000;
      errorThreshold = 2E-4;
      showInputTable = false;
      showTruthTable = true;
      numCases = 4;
      choosingWeights = "RANDOM";
      saveWeights = true;
      fileNameLoad = "inputWeights.bin";
      fileNameSave = "trainedWeights.bin";
      fileNameTruth = "truthTable.txt";
      fileNameInput = "inputTable.txt";
   } // public static void setConfigParams()

/*
 * Load configuration parameters from a text file.
 * Each line in the file corresponds to a specific parameter in the following order:
 * 0: numLayers
 * 1: numInput
 * 2: numHiddenK
 * 3: numHiddenJ
 * 4: numOutput
 * 5: modeTrain
 * 6: randLow
 * 7: randHigh
 * 8: lambda
 * 9: maxIters
 * 10: errorThreshold
 * 11: showInputTable
 * 12: showTruthTable
 * 13: numCases
 * 14: choosingWeights
 * 15: saveWeights
 * 16: fileNameLoad
 * 17: fileNameSave
 * 18: fileNameTruth
 * 19: fileNameInput
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
               case IDX_NUM_LAYERS:  numLayers = Integer.parseInt(value); break;
               case IDX_NUM_INPUT:  numInput = Integer.parseInt(value); break;
               case IDX_NUM_HIDDENK:  numHiddenK = Integer.parseInt(value); break;
               case IDX_NUM_HIDDENJ:  numHiddenJ = Integer.parseInt(value); break;
               case IDX_NUM_OUTPUT:  numOutput = Integer.parseInt(value); break;
               case IDX_MODE_TRAIN:  modeTrain = Boolean.parseBoolean(value); break;
               case IDX_RAND_LOW:  randLow = Double.parseDouble(value); break;
               case IDX_RAND_HIGH:  randHigh = Double.parseDouble(value); break;
               case IDX_LAMBDA:  lambda = Double.parseDouble(value); break;
               case IDX_MAX_ITERS:  maxIters = Integer.parseInt(value); break;
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
               default: break;
            } // switch (lineCount)

            lineCount++;
         } // while ((line = br.readLine()) != null)

         numConnectivityLayers = numLayers - NUMINPUTLAYERS;
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
      System.out.println("Number of Layers: " + numLayers);
      System.out.println("Network Configuration: " + numInput + "-" + numHiddenK + "-" + numHiddenJ + "-" + numOutput);
      System.out.println("Mode: " + (modeTrain ? "TRAIN" : "RUN"));
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
         System.out.println(String.format("Average error threshold: %.8f", errorThreshold));
         System.out.println("Truth Table File: " + fileNameTruth);
      }

      System.out.println("Input Table File: " + fileNameInput);
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
      a = new double[numLayers][];
      a[INPUTLAYER] = new double[numInput];               // input activations a[m]
      a[HIDDENLAYERK] = new double[numHiddenK];           // hidden activations a[k]
      a[HIDDENLAYERJ] = new double[numHiddenJ];           // hidden activations a[j]
      a[OUTPUTLAYER] = new double[numOutput];             // output activations a[i]

      w = new double[numLayers][][];               
      w[HIDDENLAYERK] = new double[numInput][numHiddenK];   // weights w[m][k]
      w[HIDDENLAYERJ] = new double[numHiddenK][numHiddenJ]; // weights w[k][j]
      w[OUTPUTLAYER] = new double[numHiddenJ][numOutput];   // weights w[j][i]
      
      outputs = new double[numCases][numOutput];
      testCases = new double[numCases][numInput];

      if (modeTrain)                           // Training-only arrays: allocate only when training
      {
         theta = new double[numConnectivityLayers][];    // avoiding wasted memory for the output layer
         theta[HIDDENLAYERK] = new double[numHiddenK];
         theta[HIDDENLAYERJ] = new double[numHiddenJ];
         psi = new double[numLayers][];
         psi[HIDDENLAYERJ] = new double[numHiddenJ];
         psi[OUTPUTLAYER] = new double[numOutput];
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
      dos.writeInt(numInput);
      dos.writeInt(numHiddenK);
      dos.writeInt(numHiddenJ);
      dos.writeInt(numOutput);

      for (int m = 0; m < numInput; m++) // save the weights in hidden layer 1
      {
         for (int k = 0; k < numHiddenK; k++)
         {
            dos.writeDouble(w[HIDDENLAYERK][m][k]);
         }
      }

      for (int k = 0; k < numHiddenK; k++) // save the weights in hidden layer 2
      {
         for (int j = 0; j < numHiddenJ; j++)
         {
            dos.writeDouble(w[HIDDENLAYERJ][k][j]);
         }
      }

      for (int j = 0; j < numHiddenJ; j++) // save the output weights
      {
         for (int i = 0; i < numOutput; i++)
         {
            dos.writeDouble(w[OUTPUTLAYER][j][i]);
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
      int fileNumInput = dis.readInt();
      int fileNumHiddenK = dis.readInt();
      int fileNumHiddenJ = dis.readInt();
      int fileNumOutput = dis.readInt();

      if (fileNumInput != numInput || fileNumHiddenK != numHiddenK || fileNumHiddenJ != numHiddenJ || fileNumOutput != numOutput)
      {
         throw new IllegalStateException(
               String.format("Weight file config %d-%d-%d-%d does not match current config %d-%d-%d-%d",
                  fileNumInput, fileNumHiddenK, fileNumHiddenJ, fileNumOutput, numInput, numHiddenK, numHiddenJ, numOutput));
      }

      for (int m = 0; m < numInput; m++) // load the weights for hidden layer 1
      {
         for (int k = 0; k < numHiddenK; k++)
         {
            w[HIDDENLAYERK][m][k] = dis.readDouble();
         }
      }

      for (int k = 0; k < numHiddenK; k++) // load the weights for hidden layer 2
      {
         for (int j = 0; j < numHiddenJ; j++)
         {
            w[HIDDENLAYERJ][k][j] = dis.readDouble();
         }
      }

      for (int j = 0; j < numHiddenJ; j++) // load the output weights
      {
         for (int i = 0; i < numOutput; i++)
         {
            w[OUTPUTLAYER][j][i] = dis.readDouble();
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
      setInputTable(fileNameInput);

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
* Manual weight initializer that works for any A-B-C-D.
* If (A,B) match a known case, we load hand-crafted weights.
* Otherwise, we fall back to small deterministic values.
*/
   public static void setWeightsManually()
   {
      if (numInput == 2 && numHiddenK == 2)            // if we have a 2-2-C-D network
      {
         w[HIDDENLAYERJ][0][0] = 1.0; w[HIDDENLAYERJ][1][0] = 1.0;          // k=0,1 -> j=0
         w[HIDDENLAYERJ][0][1] = -1.0; w[HIDDENLAYERJ][1][1] = 1.0;         // k=0,1 -> j=1
         w[OUTPUTLAYER][0][0] = 1.0; w[OUTPUTLAYER][1][0] = 1.0;
         w[OUTPUTLAYER][0][1] = -1.0; w[OUTPUTLAYER][1][1] = 1.0;
      } // if (numInput == 2 && numHiddenK == 2)
      else if (numInput == 2 && numHiddenK == 3)       // if we have a 2-3-C-D network
      {
         w[HIDDENLAYERJ][0][0] = 5.0;  w[HIDDENLAYERJ][1][0] = 5.0;
         w[HIDDENLAYERJ][0][1] = -5.0; w[HIDDENLAYERJ][1][1] = 5.0;
         w[HIDDENLAYERJ][0][2] = 5.0;  w[HIDDENLAYERJ][1][2] = -5.0;
         w[OUTPUTLAYER][0][0] = 5.0; w[OUTPUTLAYER][1][0] = -5.0;
         w[OUTPUTLAYER][0][1] = -5.0; w[OUTPUTLAYER][1][1] = 5.0;
         w[OUTPUTLAYER][0][2] = 5.0; w[OUTPUTLAYER][1][2] = 5.0;
      } // else if (numInput == 2 && numHiddenK == 3)
      else if (numInput == 2 && numHiddenK == 5)       // if we have a 2-5-C-D network
      {
         w[HIDDENLAYERJ][0][0] = 20.0; w[HIDDENLAYERJ][1][0] = 20.0;
         w[HIDDENLAYERJ][0][1] = -20.0; w[HIDDENLAYERJ][1][1] = -20.0;
         w[HIDDENLAYERJ][0][2] = 20.0; w[HIDDENLAYERJ][1][2] = -20.0;
         w[HIDDENLAYERJ][0][3] = -20.0; w[HIDDENLAYERJ][1][3] = 20.0;
         w[HIDDENLAYERJ][0][4] = 20.0; w[HIDDENLAYERJ][1][4] = 20.0;
         w[OUTPUTLAYER][0][0] = -20.0; w[OUTPUTLAYER][1][0] = -20.0;
         w[OUTPUTLAYER][0][1] = 20.0; w[OUTPUTLAYER][1][1] = 20.0;
         w[OUTPUTLAYER][0][2] = -20.0; w[OUTPUTLAYER][1][2] = -20.0;
         w[OUTPUTLAYER][0][3] = 20.0; w[OUTPUTLAYER][1][3] = 20.0;
      } // else if (numInput == 2 && numHiddenK == 5)
      else                                            // set the weights to small deterministic values
      {
         for (int m = 0; m < numInput; m++)
         {
            for (int k = 0; k < numHiddenK; k++)
            {
               w[HIDDENLAYERK][m][k] = 0.1 * (k + 1) + 0.05 * (m + 1);
            }
         }

         for (int k = 0; k < numHiddenK; k++)
         {
            for (int j = 0; j < numHiddenJ; j++)
            {
               w[HIDDENLAYERJ][k][j] = 0.1 * (j + 1) + 0.05 * (k + 1);
            }
         }
         
         for (int j = 0; j < numHiddenJ; j++)
         {
            for (int i = 0; i < numOutput; i++)
            {
               w[OUTPUTLAYER][j][i] = 0.1 * (j + 1);
            }
         }
      } // else
   } // public static void setWeightsManually()

/*
* Select random weights for the network, with each weight ranging from the low to high values passed in as parameters.
* Calls the randRange method below to select a random value for each weight in the arrays.
* @param low the lowest possible value for the weights
* @param high the highest possible value for the weights
*/
   public static void randomizeWeights(double low, double high)
   {
      for (int m = 0; m < numInput; m++)
      {
         for (int k = 0; k < numHiddenK; k++)
         {
            w[HIDDENLAYERK][m][k] = randRange(low, high);
         }
      }

      for (int k = 0; k < numHiddenK; k++)
      {
         for (int j = 0; j < numHiddenJ; j++)
         {
            w[HIDDENLAYERJ][k][j] = randRange(low, high);
         }
      }
      
      for (int j = 0; j < numHiddenJ; j++)
      {
         for (int i = 0; i < numOutput; i++)
         {
            w[OUTPUTLAYER][j][i] = randRange(low, high);
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
      for (int m = 0; m < numInput; m++)
      {
         a[INPUTLAYER][m] = input[m]; // copy inputs into a[m]
      }
   }

/*
* Run for training. This method only runs the network once.
* It does not depend on truth table and does no training allocations.
* theta is not local (it is an array).
* Coded directly to the design document.
*/
   public static double runForTraining(int n)
   {
/*
 * Compute Θ_k and a[k] for the mk layer
 */
      for (int k = 0; k < numHiddenK; k++)
      {
         double sum = 0.0;

         for (int m = 0; m < numInput; m++)
         {
            sum += a[INPUTLAYER][m] * w[HIDDENLAYERK][m][k];
         }

         theta[HIDDENLAYERK][k] = sum;
         a[HIDDENLAYERK][k] = activation(theta[HIDDENLAYERK][k]);
      } // for (int k = 0; k < numHiddenK; k++)
/*
* Compute Θ_j and a[j] for the kj layer
*/ 
      for (int j = 0; j < numHiddenJ; j++)
      {
         double sum = 0.0;

         for (int k = 0; k < numHiddenK; k++)
         {
            sum += a[HIDDENLAYERK][k] * w[HIDDENLAYERJ][k][j];
         }

         theta[HIDDENLAYERJ][j] = sum;
         a[HIDDENLAYERJ][j] = activation(theta[HIDDENLAYERJ][j]);
      } // for (int j = 0; j < numHiddenJ; j++)
/*
* Compute Θ_i and F for the ji layer
*/
      double sumError = 0.0;

      for (int i = 0; i < numOutput; i++)
      {
         double theta_i = 0.0;

         for (int j = 0; j < numHiddenJ; j++)
         {
            theta_i += a[HIDDENLAYERJ][j] * w[OUTPUTLAYER][j][i];
         }

         a[OUTPUTLAYER][i] = activation(theta_i);
         double omega_i = truth[n][i] - a[OUTPUTLAYER][i];
         psi[OUTPUTLAYER][i] = omega_i * activationDeriv(theta_i);
         sumError += omega_i * omega_i;
      } // for (int i = 0; i < numOutput; i++)
      
      sumError /= 2.0;
      return sumError;
   } // public static double runForTraining(int n)

/*
* Run for running. This method only runs the network once.
* It does not depend on truth table and does no training allocations.
* Additionally, theta_j is local.
* Coded directly to the design document.
*/
   public static void runForRunning(int n)
   {
/*
 * Compute Θ_k and a[k] for the mk layer
 */
      for (int k = 0; k < numHiddenK; k++)
      {
         double theta_k = 0.0;

         for (int m = 0; m < numInput; m++)
         {
            theta_k += a[INPUTLAYER][m] * w[HIDDENLAYERK][m][k];
         }

         a[HIDDENLAYERK][k] = activation(theta_k);
      } // for (int k = 0; k < numHiddenK; k++)
/*
* Compute Θ_j and a[j] for the kj layer
*/ 
      for (int j = 0; j < numHiddenJ; j++)
      {
         double theta_j = 0.0;

         for (int k = 0; k < numHiddenK; k++)
         {
            theta_j += a[HIDDENLAYERK][k] * w[HIDDENLAYERJ][k][j];
         }

         a[HIDDENLAYERJ][j] = activation(theta_j);
      } // for (int j = 0; j < numHiddenJ; j++)
/*
* Compute Θ_i and F for the ji layer
*/
      for (int i = 0; i < numOutput; i++)
      {
         double theta_i = 0.0;

         for (int j = 0; j < numHiddenJ; j++)
         {
            theta_i += a[HIDDENLAYERJ][j] * w[OUTPUTLAYER][j][i];
         }

         a[OUTPUTLAYER][i] = activation(theta_i);
      } // for (int i = 0; i < numOutput; i++)
   } // public static void runForRunning(int n)

/*
 * Set up the input activations for training the network.
 * @param n the index of the test case to set the activations from
 * @postcondition a[k] has been set to the input activations for test case n.
 */
   public static void setUpTrainActivations(int n)
   {
      for (int m = 0; m < numInput; m++)
      {
         a[INPUTLAYER][m] = testCases[n][m];
      }
   }

/*
* Train (gradient descent). Training is implemented exactly as the algorithm
* in the design document.
*
* @precondition in training mode; truth and F are populated.
*/
   public static void trainOnce()
   {
      for (int j = 0; j < numHiddenJ; j++)
      {
         double omega_j = 0.0;

         for (int i = 0; i < numOutput; i++)
         {
            omega_j += psi[OUTPUTLAYER][i] * w[OUTPUTLAYER][j][i];
            w[OUTPUTLAYER][j][i] += lambda * a[HIDDENLAYERJ][j] * psi[OUTPUTLAYER][i]; // update weights (output)
         }

         psi[HIDDENLAYERJ][j] = omega_j * activationDeriv(theta[HIDDENLAYERJ][j]);
      } // for (int j = 0; j < numHiddenJ; j++)

      for (int k = 0; k < numHiddenK; k++)
      {
         double omega_k = 0.0;

         for (int j = 0; j < numHiddenJ; j++)
         {
            omega_k += psi[HIDDENLAYERJ][j] * w[HIDDENLAYERJ][k][j];
            w[HIDDENLAYERJ][k][j] += lambda * a[HIDDENLAYERK][k] * psi[HIDDENLAYERJ][j]; // update weights (hidden)
         }

         double psi_k = omega_k * activationDeriv(theta[HIDDENLAYERK][k]);
         
         for (int m = 0; m < numInput; m++)
         {
            w[HIDDENLAYERK][m][k] += lambda * a[INPUTLAYER][m] * psi_k; // update weights (input)
         }
      } // for (int k = 0; k < numHiddenK; k++)
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
         runForRunning(t); // run the network

         for (int i = 0; i < numOutput; i++)
         {
            outputs[t][i] = a[OUTPUTLAYER][i];
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
            for (int m = 0; m < numInput; m++) System.out.print((int) testCases[t][m] + " ");
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
         System.out.print(String.format("Input: %.4f %.4f -> Output: ", 
         testCases[t][0], testCases[t][1]));

         for (int i = 0; i < numOutput; i++)
         {
            System.out.print(String.format("%.4f ", outputs[t][i]));
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
} // public class ABCDBackprop