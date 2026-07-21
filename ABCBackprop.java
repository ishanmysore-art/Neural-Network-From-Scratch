import java.time.LocalDateTime;
import java.util.Random;
import java.io.*;

/*
 * @author Ishan Mysore
 * @date 10/9/25
 * @file ABCBackprop.java
 * 
 * This code implements the backpropagation training algorithm for training neural networks in Java, restricted to an 
 * A-B-C architecture (A inputs, B hidden nodes, C outputs). The code separates functionality into different methods, 
 * including configuration, memory allocation, weight initialization, training, and running the network. The system supports
 * independent runtime modes for training and running, outputs full configuration and results, and handles user-selectable weight 
 * initialization (manual, randomized, or file input). Finally, this code demonstrates functionality on standard Boolean test
 * cases (AND, OR, XOR) with specified parameters.
 * 
 * This code also adds how much time it took to perform the training/running along with the hyperbolic tangent threshold function.
 * Additionally, the code implements the use of a control file to tell the program what to do at execution time.
 * Lastly, this code implements saving and loading weights to/from a binary file.
 * All features are implemented as per the project specification and design document.
 */
public class ABCBackprop
{
/*
* Configuration parameters (loaded from configuration file)
*/
   public static int numInput;            // number of input activations (a[k])
   public static int numHidden;           // number of hidden activations (h[j])
   public static int numOutput;           // number of output activations (F[i])
   public static boolean modeTrain;       // true = training; false = running
   public static boolean showInputTable;  // option for run report
   public static boolean showTruthTable;  // option for run report (running does not require truth table)
   public static double randLow;          // lowest value for random generation
   public static double randHigh;         // highest value for random generation
   public static double lambda;           // learning rate
   public static int maxIters;            // maximum number of iterations that the network will train on
   public static double errorThreshold;   // average error threshold (stop when average error <= this value)
   public static int numCases;            // for boolean problems: 4 cases 00,01,10,11 in this order
   public static String activation;       // the activation function (sigmoid, hyperbolic tangent, etc)
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
   private static final int IDX_NUM_INPUT = 0;
   private static final int IDX_NUM_HIDDEN = 1;
   private static final int IDX_NUM_OUTPUT = 2;
   private static final int IDX_MODE_TRAIN = 3;
   private static final int IDX_RAND_LOW = 4;
   private static final int IDX_RAND_HIGH = 5;
   private static final int IDX_LAMBDA = 6;
   private static final int IDX_MAX_ITERS = 7;
   private static final int IDX_ERROR_THRESHOLD = 8;
   private static final int IDX_SHOW_INPUT_TABLE = 9;
   private static final int IDX_SHOW_TRUTH_TABLE = 10;
   private static final int IDX_NUM_CASES = 11;
   private static final int IDX_ACTIVATION = 12;
   private static final int IDX_CHOOSING_WEIGHTS = 13;
   private static final int IDX_SAVE_WEIGHTS = 14;
   private static final int IDX_FILE_NAME_LOAD = 15;
   private static final int IDX_FILE_NAME_SAVE = 16;
   private static final int IDX_FILE_NAME_TRUTH = 17;
   private static final int IDX_FILE_NAME_INPUT = 18;
/*
* Network arrays (allocated in allocateMemory)
*/
   public static double[] a;             // a[k] input activations
   public static double[] h;             // h[j] hidden activations
   public static double[] F;             // F[i] output activations
   public static double[][] w_kj;        // w_kj: input k -> hidden j as [numInput][numHidden]
   public static double[][] w_ji;        // w_ji: hidden j -> output activations
   public static double[] theta_j;       // Θ_j (net into hidden j)     
   public static double[] psi_i;         // ψ_i = w_i * f'(Θ_i)
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
      numInput = 2;
      numHidden = 3;
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
      activation = "SIGMOID";
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
 * 0: numInput (int)
 * 1: numHidden (int)
 * 2: numOutput (int)
 * 3: modeTrain (boolean)
 * 4: randLow (double)
 * 5: randHigh (double)
 * 6: lambda (double)
 * 7: maxIters (int)
 * 8: errorThreshold (double)
 * 9: showInputTable (boolean)
 * 10: showTruthTable (boolean)
 * 11: numCases (int)
 * 12: activation (String)
 * 13: choosingWeights (String)
 * 14: saveWeights (boolean)
 * 15: fileNameLoad (String)
 * 16: fileNameSave (String)
 * 17: fileNameTruth (String)
 * 18: fileNameInput (String)
 * @postcondition all configuration parameters have been set from the file.
 */
   public static void loadConfigParams()
   {
      try (BufferedReader br = new BufferedReader(new FileReader("configFile.txt")))
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
               case IDX_NUM_INPUT:  numInput = Integer.parseInt(value); break;
               case IDX_NUM_HIDDEN:  numHidden = Integer.parseInt(value); break;
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
               case IDX_ACTIVATION: activation = value; break;
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
      } // try (BufferedReader br = new BufferedReader(new FileReader("configFile.txt")))
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
      System.out.println("Network Configuration: " + numInput + "-" + numHidden + "-" + numOutput);
      System.out.println("Mode: " + (modeTrain ? "TRAIN" : "RUN"));
      System.out.println("Populate Method: " + choosingWeights);
      System.out.println(String.format("Random range: [%.4f, %.4f]", randLow, randHigh));

      if (modeTrain)
      {
         System.out.println(String.format("Learning Rate: %.6f", lambda));
         System.out.println("Max iterations: " + maxIters);
         System.out.println(String.format("Average error threshold: %.8f", errorThreshold));
      }

      System.out.println("Show Input Table (run): " + showInputTable);
      System.out.println("Show Truth Table (run): " + showTruthTable);
   } // public static void echoConfigParams()

/*
* Allocate memory for each of the network arrays listed above.
* @postcondition all of the arrays have memory allocated
*/
   public static void allocateMemory()
   {
      a = new double[numInput];                // inputs a[k]
      h = new double[numHidden];               // hidden activations h[j]
      F = new double[numOutput];               // output activations F[i]

      w_kj = new double[numInput][numHidden];  // w_kj[k][j]
      w_ji = new double[numHidden][numOutput]; // w_ji[j][i]
      
      outputs = new double[numCases][numOutput];
      testCases = new double[numCases][numInput];

      if (modeTrain)                           // Training-only arrays: allocate only when training
      {  
         theta_j = new double[numHidden];
         truth = new double[numCases][numOutput];
         psi_i = new double[numOutput];
      } // if (modeTrain)
   } // public static void allocateMemory()

/*
 * Save the current network weights to a binary file.
 * The file format includes the network configuration (numInput, numHidden, numOutput)
 * followed by all weights in a consistent order:
 *   - w_kj[k][j] (hidden × input)
 *   - w_ji[j][i] (hidden → output)
 *
 * @param filename the binary file to save to
 */
public static void saveWeights(String filename)
{
   try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename)))
   {
      dos.writeInt(numInput);
      dos.writeInt(numHidden);
      dos.writeInt(numOutput);

      for (int k = 0; k < numInput; k++) // save the hidden weights
      {
         for (int j = 0; j < numHidden; j++)
         {
            dos.writeDouble(w_kj[k][j]);
         }
      }

      for (int j = 0; j < numHidden; j++) // save the output weights
      {
         for (int i = 0; i < numOutput; i++)
         {
            dos.writeDouble(w_ji[j][i]);
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
      int fileNumHidden = dis.readInt();
      int fileNumOutput = dis.readInt();

      if (fileNumInput != numInput || fileNumHidden != numHidden || fileNumOutput != numOutput)
      {
         throw new IllegalStateException(
               String.format("Weight file config %d-%d-%d does not match current config %d-%d-%d",
                  fileNumInput, fileNumHidden, fileNumOutput, numInput, numHidden, numOutput));
      }

      for (int k = 0; k < numInput; k++) // load hidden weights
      {
         for (int j = 0; j < numHidden; j++)
         {
            w_kj[k][j] = dis.readDouble();
         }
      }

      for (int j = 0; j < numHidden; j++) // load output weights
      {
         for (int i = 0; i < numOutput; i++)
         {
            w_ji[j][i] = dis.readDouble();
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
* Manual weight initializer that works for any A-B-C.
* If (A,B) match a known case, we load hand-crafted weights.
* Otherwise, we fall back to small deterministic values.
*/
   public static void setWeightsManually()
   {
      if (numInput == 2 && numHidden == 2)            // if we have a 2-2-C network
      {
         w_kj[0][0] = 1.0; w_kj[1][0] = 1.0;          // k=0,1 -> j=0
         w_kj[0][1] = -1.0; w_kj[1][1] = 1.0;         // k=0,1 -> j=1
         w_ji[0][0] = 1.0; w_ji[1][0] = 1.0;
         w_ji[0][1] = -1.0; w_ji[1][1] = 1.0;
      } // if (numInput == 2 && numHidden == 2)
      else if (numInput == 2 && numHidden == 3)       // if we have a 2-3-C network
      {
         w_kj[0][0] = 5.0;  w_kj[1][0] = 5.0;
         w_kj[0][1] = -5.0; w_kj[1][1] = 5.0;
         w_kj[0][2] = 5.0;  w_kj[1][2] = -5.0;
         w_ji[0][0] = 5.0; w_ji[1][0] = -5.0;
         w_ji[0][1] = -5.0; w_ji[1][1] = 5.0;
         w_ji[0][2] = 5.0; w_ji[1][2] = 5.0;
      } // else if (numInput == 2 && numHidden == 3)
      else if (numInput == 2 && numHidden == 5)       // if we have a 2-5-C network
      {
         w_kj[0][0] = 20.0; w_kj[1][0] = 20.0;
         w_kj[0][1] = -20.0; w_kj[1][1] = -20.0;
         w_kj[0][2] = 20.0; w_kj[1][2] = -20.0;
         w_kj[0][3] = -20.0; w_kj[1][3] = 20.0;
         w_kj[0][4] = 20.0; w_kj[1][4] = 20.0;
         w_ji[0][0] = -20.0; w_ji[1][0] = -20.0;
         w_ji[0][1] = 20.0; w_ji[1][1] = 20.0;
         w_ji[0][2] = -20.0; w_ji[1][2] = -20.0;
         w_ji[0][3] = 20.0; w_ji[1][3] = 20.0;
      } // else if (numInput == 2 && numHidden == 5)
      else                                            // set the weights to small deterministic values
      {
         for (int k = 0; k < numInput; k++)
         {
            for (int j = 0; j < numHidden; j++)
            {
               w_kj[k][j] = 0.1 * (j + 1) + 0.05 * (k + 1);
            }
         }
         
         for (int j = 0; j < numHidden; j++)
         {
            for (int i = 0; i < numOutput; i++)
            {
               w_ji[j][i] = 0.1 * (j + 1);
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
      for (int k = 0; k < numInput; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            w_kj[k][j] = randRange(low, high);
         }
      }
      
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutput; i++)
         {
            w_ji[j][i] = randRange(low, high);
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
 * Returns +1 if x < 0, else -1.
 * @param x the input value
 */
   public static double epsilon(double x)
   {
      return (x < 0.0) ? 1.0 : -1.0;
   }

/*
 * Calculates the hyperbolic tangent activation function for an input x.
 * @param x the input that we want to calculate the hyperbolic tangent of.
 * @return the hyperbolic tangent function for x
 */
   public static double tanh(double x)
   {
      double eps = epsilon(x);
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
      for (int k = 0; k < numInput; k++)
      {
         a[k] = input[k]; // copy inputs into a[k]
      }
   }

/*
* Run for training. This method only runs the network once.
* It does not depend on truth table and does no training allocations.
* theta_j is not local (it is an array).
* Coded directly to the design document.
*/
   public static double runForTraining(int n)
   {
/*
* compute Θ_j and h[j]
*/ 
      for (int j = 0; j < numHidden; j++)
      {
         double sum = 0.0;

         for (int k = 0; k < numInput; k++)
         {
            sum += a[k] * w_kj[k][j]; // w_kj[k][j]
         }

         theta_j[j] = sum;
         h[j] = activation(theta_j[j]);
      } // for (int j = 0; j < numHidden; j++)
/*
* compute Θ_i and F
*/
      double sumError = 0.0;

      for (int i = 0; i < numOutput; i++)
      {
         double theta_i = 0.0;

         for (int j = 0; j < numHidden; j++)
         {
            theta_i += h[j] * w_ji[j][i];
         }

         F[i] = activation(theta_i);
         double omega_i = truth[n][i] - F[i];
         psi_i[i] = omega_i * activationDeriv(theta_i);
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
* compute Θ_j and h[j]
*/ 
      for (int j = 0; j < numHidden; j++)
      {
         double theta_j = 0.0;

         for (int k = 0; k < numInput; k++)
         {
            theta_j += a[k] * w_kj[k][j];
         }

         h[j] = activation(theta_j);
      } // for (int j = 0; j < numHidden; j++)
/*
* compute Θ_i and F
*/
      for (int i = 0; i < numOutput; i++)
      {
         double theta_i = 0.0;

         for (int j = 0; j < numHidden; j++)
         {
            theta_i += h[j] * w_ji[j][i];
         }

         F[i] = activation(theta_i);
      } // for (int i = 0; i < numOutput; i++)
   } // public static void runForRunning(int n)

/*
 * Set up the input activations for training the network.
 * @param n the index of the test case to set the activations from
 * @postcondition a[k] has been set to the input activations for test case n.
 */
   public static void setUpTrainActivations(int n)
   {
      for (int k = 0; k < numInput; k++)
      {
         a[k] = testCases[n][k];
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
      for (int j = 0; j < numHidden; j++)
      {
         double omega_j = 0.0;

         for (int i = 0; i < numOutput; i++)
         {
            omega_j += psi_i[i] * w_ji[j][i];
            double delta_w_ji = lambda * h[j] * psi_i[i];
            w_ji[j][i] += delta_w_ji;     // update weights (output)
         }

         double psi_j = omega_j * activationDeriv(theta_j[j]);

         for (int k = 0; k < numInput; k++)
         {
            double delta_w_kj = lambda * a[k] * psi_j;
            w_kj[k][j] += delta_w_kj;     // update weights (hidden)
         }
      }
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
         System.out.println(String.format("Average error: %.8f", avgError));
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
            outputs[t][i] = F[i];
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
            for (int k = 0; k < numInput; k++) System.out.print((int) testCases[t][k] + " ");
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
         }
      } // if (showTruthTable && truth != null)
/*
* Prints the output values.
*/
      System.out.println("Actual Outputs:");
      for (int t = 0; t < numCases; t++)
      {
         System.out.print(String.format("Input: %d %d -> Output: ", 
         (int) testCases[t][0], (int) testCases[t][1]));

         for (int i = 0; i < numOutput; i++)
         {
            System.out.print(String.format("%.17f ", outputs[t][i]));
         }

         System.out.println();
      }
      System.out.println("--------------------");
   } // public static void printResults()

/*
* Puts it all together!
*/
   public static void main(String[] args)
   {
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
         if (saveWeights) saveWeights(fileNameLoad);
      }     

      runAllTestCases();
      endTime = System.currentTimeMillis();
      printResults();
   } // public static void main(String[] args)
} // public class ABCBackprop