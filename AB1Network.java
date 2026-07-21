import java.time.LocalDateTime;
import java.util.Random;

/**
 * @author Ishan Mysore
 * @date 9/2/25
 * @file AB1Network.java
 * 
 * This code implements the steepest descent training algorithm for training neural networks in Java, restricted to an 
 * A-B-1 architecture (A inputs, B hidden nodes, 1 output). The code separates functionality into different methods, 
 * including configuration, memory allocation, weight initialization, training, and running the network. The system supports
 * independent runtime modes for training and running, outputs full configuration and results, and handles user-selectable weight 
 * initialization (manual or randomized). Finally, this code demonstrates functionality on standard Boolean test cases
 * (AND, OR, XOR) with specified parameters and shows that the network can run with either pre-trained or random weights.
 *
 * All other documentation, method names, and coding style have been preserved.
 */
public class AB1Network
{
/*
* Configuration parameters (set in setConfigParams)
*/
   public static int numInput;            // number of input activations (a[k])
   public static int numHidden;           // number of hidden activations (h[j])
   public static boolean modeTrain;       // true = training; false = running
   public static boolean populateRandom;  // if false, use manual weights (if provided)
   public static boolean showInputTable;  // option for run report
   public static boolean showTruthTable;  // option for run report (running does not require truth table)
   public static double randLow;          // lowest value for random generation
   public static double randHigh;         // highest value for random generation
   public static double lambda;           // learning rate
   public static int maxIters;            // maximum number of iterations that the network will train on
   public static double errorThreshold;   // average error threshold (stop when average error <= this value)
   public static boolean preloadWeights;  // if true, populate with hard-coded weights
   public static int nCases;              // for boolean problems: 4 cases 00,01,10,11 in this order
   public static String boolProb;         // AND, OR, XOR
   public static String activation;       // the activation function (sigmoid, hyperbolic tangent, etc)
/*
* Network arrays (allocated in allocateMemory)
*/
   public static double[] a;             // a[k] input activations
   public static double[] h;             // h[j] hidden activations
   public static double[][] w_kj;        // w_kj: input k -> hidden j as [numInput][numHidden]
   public static double[] w_j0;          // w_j0: hidden j -> output
   public static double[] theta_j;       // Θ_j (net into hidden j)
   public static double[][] delta_w_kj;  // stored Δw_kj [numInput][numHidden]
   public static double[] delta_w_j0;    // stored Δw_j0 [numHidden]
   public static double[] omega_j;       // Ω_j = ψ_0 * w_j0
   public static double[] psi_j;         // ψ_j = Ω_j * f'(Θ_j)
   public static double[][] testCases;   // size [nCases][numInput]
   public static double[] truth;         // truth table (targets) size [nCases] (training-only)
   public static double[] outputs;       // the values that the neural network outputs after running
/*
* Output parameters (for training only)
*/
   public static double omega_0;          // ω_0 = (T0 - F0)
   public static double psi_0;            // ψ_0 = ω_0 * f'(Θ_0)
   public static double theta_0;          // Θ_0 (net into output)
   public static double F0;               // F_0 output activation
   public static double avgError;         // average training error
   public static int iterations;          // number of training iterations
   public static boolean reachedError;    // tracks whether we are under the error threshold
   public static boolean reachedMaxIter;  // tracks whether we have reached the max number of iterations
/*
* Other miscellaneous parameters
*/
   public static Random rng;              // Random Number Generator

/*
* Set hard-coded values for each of the configuration parameters listed above.
* @postcondition all the configuration parameters have been set to user values.
*/
   public static void setConfigParams()
   {
      numInput = 2;
      numHidden = 2;
      boolProb = "XOR";
      modeTrain = true;
      populateRandom = true;
      randLow = -1.5;
      randHigh = 1.5;
      lambda = 0.3;
      maxIters = 100000;
      errorThreshold = 2E-4;
      preloadWeights = false;
      showInputTable = false;
      showTruthTable = true;
      nCases = 4;
      activation = "SIGMOID";
   } // public static void setConfigParams()

/*
* Initialize the random number generator.
* @postcondition the random number generator has been initialized.
*/
   public static void initializeRNG()
   {
      rng = new Random(); // initialize Random Number Generator
   }

/*
* Print the configuration parameters in a readable format for the user.
*/
   public static void echoConfigParams()
   {
      System.out.println("Time: " + LocalDateTime.now());   // print the current date and time
      System.out.println("=======================================================================================================");
      System.out.println("--- Configuration Parameters ---");
      System.out.println("Network Configuration: " + numInput + "-" + numHidden + "-1");     // output the network configuration
      System.out.println("Mode: " + (modeTrain ? "TRAIN" : "RUN"));                          // output which mode we are in (training or running)
      System.out.println("Boolean Problem: " + boolProb);
      System.out.println("Populate Method: " + (populateRandom ? "Randomize" : "Manual/Preloaded"));
      System.out.println(String.format("Random range: [%.4f, %.4f]", randLow, randHigh));
      if (modeTrain)
      {
         System.out.println(String.format("Learning Rate: %.6f", lambda));
         System.out.println("Max iterations: " + maxIters);
         System.out.println(String.format("Average error threshold: %.8f", errorThreshold));
      }
      System.out.println("Use preloaded weights: " + preloadWeights);
      System.out.println("Show Input Table (run): " + showInputTable);
      System.out.println("Show Truth Table (run): " + showTruthTable);
   } // public static void echoConfigParams()

/*
* Allocate memory for each of the network arrays listed above.
* @postcondition all of the arrays have memory allocated
*/
   public static void allocateMemory()
   {
      a = new double[numInput];      // inputs a[k]
      h = new double[numHidden];     // hidden activations h[j]

      w_kj = new double[numInput][numHidden];  // w_kj[k][j]
      w_j0 = new double[numHidden];            // w_j0[j]

      theta_j = new double[numHidden];
      outputs = new double[nCases];
      
      testCases = new double[nCases][numInput];
      for (int j = 0; j < nCases; j++) testCases[j] = new double[numInput];

      if (modeTrain) // Training-only arrays: allocate only when training
      {
         truth = new double[nCases];
         omega_j = new double[numHidden];
         psi_j = new double[numHidden];
         delta_w_kj = new double[numInput][numHidden];
         delta_w_j0 = new double[numHidden];
      }
      else
      {
/*
* do NOT allocate T[] in run-only mode
*/
         truth = null;
         omega_j = null;
         psi_j = null;
         delta_w_kj = null;
         delta_w_j0 = null;
      }
   } // public static void allocateMemory()

/**
 * Populate values for each of the arrays for which memory was allocated earlier.
* Also set the weights manually/randomize them, depending on the corresponding parameters.
* If both preloadWeights and populateRandom are false, set the weights to zero.
*/
   public static void populateArrays()
   {
      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;
      testCases[1][0] = 0.0;
      testCases[1][1] = 1.0;
      testCases[2][0] = 1.0;
      testCases[2][1] = 0.0;
      testCases[3][0] = 1.0;
      testCases[3][1] = 1.0;

      if (modeTrain)
      {
/*
* populate truth table according to requested boolean problem
*/
         if (boolProb.equalsIgnoreCase("OR"))
         {
            truth[0] = 0.0;
            truth[1] = 1.0;
            truth[2] = 1.0;
            truth[3] = 1.0;
         }
         else if (boolProb.equalsIgnoreCase("AND"))
         {
            truth[0] = 0.0;
            truth[1] = 0.0;
            truth[2] = 0.0;
            truth[3] = 1.0;
         }
         else // XOR
         {
            truth[0] = 0.0;
            truth[1] = 1.0;
            truth[2] = 1.0;
            truth[3] = 0.0;
         }
      } // if (modeTrain)

/*
* If preloadWeights is true, set the weights manually.
*/
      if (preloadWeights)
      {
         setWeightsManually();
         return;
      } // if (preloadWeights)

/*
* If populateRandom is true, randomize the weights.
*/
      if (populateRandom)
      {
         randomizeWeights(randLow, randHigh);
      }
      else // Set all the weights to zero otherwise.
      {
         for (int k = 0; k < numInput; k++)
         {
            for (int j = 0; j < numHidden; j++)
            {
               w_kj[k][j] = 0.0;
            }
         }
         for (int j = 0; j < numHidden; j++) w_j0[j] = 0.0;
      } // else
   } // public static void populateArrays()

/**
* Manual weight initializer that works for any A-B-1.
* If (A,B) match a known case (like 2-2-1 OR), we load hand-crafted weights.
* Otherwise, we fall back to small deterministic values.
*/
   public static void setWeightsManually()
   {
      if (numInput == 2 && numHidden == 2)   // if we have a 2-2-1 network
      {
         w_kj[0][0] = 1.0; w_kj[1][0] = 1.0;   // k=0,1 -> j=0
         w_kj[0][1] = -1.0; w_kj[1][1] = 1.0;  // k=0,1 -> j=1
         w_j0[0] = 1.0; w_j0[1] = 1.0;
      }
      else if (numInput == 2 && numHidden == 3)    // if we have a 2-3-1 network
      {
         w_kj[0][0] = 5.0;  w_kj[1][0] = 5.0;
         w_kj[0][1] = -5.0; w_kj[1][1] = 5.0;
         w_kj[0][2] = 5.0;  w_kj[1][2] = -5.0;
         w_j0[0] = 5.0; w_j0[1] = -5.0; w_j0[2] = -5.0;
      }
      else if (numInput == 2 && numHidden == 5)    // if we have a 2-5-1 network
      {
         w_kj[0][0] = 20.0; w_kj[1][0] = 20.0;
         w_kj[0][1] = -20.0; w_kj[1][1] = -20.0;
         w_kj[0][2] = 20.0; w_kj[1][2] = -20.0;
         w_kj[0][3] = -20.0; w_kj[1][3] = 20.0;
         w_kj[0][4] = 20.0; w_kj[1][4] = 20.0;
         w_j0[0] = 20.0; w_j0[1] = 20.0; w_j0[2] = 20.0; w_j0[3] = 20.0; w_j0[4] = -20.0;
      } // else if (numInput == 2 && numHidden == 5)    // if we have a 2-5-1 network
      else     // set the weights to small deterministic values
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
            w_j0[j] = 0.1 * (j + 1);
         }
      } // else
   } // public static void setWeightsManually()

/**
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
      for (int j = 0; j < numHidden; j++) w_j0[j] = randRange(low, high);
   } // public static void randomizeWeights(double low, double high)

/**
* Select a random number from low to high.
* @param low the lowest possible value we want to select
* @param high the highest possible value we want to select
* @return the random value generated
*/
   public static double randRange(double low, double high)
   {
      return low + rng.nextDouble() * (high - low);
   }

/**
* Calculates the activation function (sigmoid) for an input x.
* @param x the input that we want to calculate the sigmoid of.
* @return the sigmoid function for x
*/
   public static double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/**
* Calculates the derivative of the sigmoid function for an input x.
* @param x the input that we want to calculate the sigmoid of.
* @return the derivative of the sigmoid function for x
*/
   public static double sigmoidDeriv(double x)
   {
      double sig = sigmoid(x);
      return sig * (1.0 - sig);
   }

   public static double activation(double x)
   {
      return sigmoid(x);
   }

   public static double activationDeriv(double x)
   {
      return sigmoidDeriv(x);
   }

   public static void setUpRunActivations(double[] input)
   {
      for (int k = 0; k < numInput; k++)
      {
         a[k] = input[k]; // copy inputs into a[k]
      }
   }

/**
* Run (single forward pass). This method only runs the network once.
* It does not depend on T[] and does no training allocations.
* Coded directly to the design document:
*   Θ_j = Σ_k a_k w_kj
*   h_j = f(Θ_j)
*   Θ_0 = Σ_j h_j w_j0
*   F0 = f(Θ_0)
*
* @param input the array of input activations
*/
   public static double runOnce()
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
* compute Θ_0 and F0
*/
      double sum0 = 0.0;
      for (int j = 0; j < numHidden; j++)
      {
         sum0 += h[j] * w_j0[j];
      }
      theta_0 = sum0;
      F0 = activation(theta_0);
      return F0;
   } // public static void runOnce(double[] input)

   public static void setUpActivations(int n)
   {
      for (int k = 0; k < numInput; k++)
      {
         a[k] = testCases[n][k]; // a[k] = input; setting input activations
      }
   }

/**
* Train (gradient descent). Training is implemented exactly as the four-step algorithm
* in the design document.
*
* Steps for each training case n:
*   Step 1: compute ω_0 and case error
*   Step 2: compute ψ_0, Ω_j, ψ_j
*   Step 3: compute gradients ∂E/∂w and store Δw = -λ ∂E/∂w, then apply stored Δw in final loops
*
* @precondition in training mode; testCases and T are populated.
*/
   public static double trainOnce(int n, double sumError)
   {
/**
* Step 1: Compute the error
*/
      omega_0 = truth[n] - F0;                        // ω_0 = (T0 - F0)
      double caseError = 0.5 * omega_0 * omega_0;
      sumError += caseError;
/**
* Step 2: Compute the psis
*/
      psi_0 = omega_0 * activationDeriv(theta_0);              // output; ψ_0 = ω_0 * f'(Θ_0)

      for (int j = 0; j < numHidden; j++)                      // hidden psis
      {
         omega_j[j] = psi_0 * w_j0[j];                         // Ω_j = ψ_0 * w_j0
         psi_j[j] = omega_j[j] * activationDeriv(theta_j[j]);  // ψ_j = Ω_j * f'(Θ_j)
      }
/**
* Step 3: Compute Δw
*/
      for (int j = 0; j < numHidden; j++)
      {
         delta_w_j0[j] = lambda * h[j] * psi_0;          // output Δw
      }
      for (int k = 0; k < numInput; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            delta_w_kj[k][j] = lambda * a[k] * psi_j[j]; // hidden Δw
         }
      }
      for (int j = 0; j < numHidden; j++)
      {
         w_j0[j] += delta_w_j0[j];                       // update weights (output)
      }
      for (int k = 0; k < numInput; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            w_kj[k][j] += delta_w_kj[k][j];              // update weights (hidden)
         }
      }
      return sumError;
   } // public static double trainOnce(int n, double sumError)

   public static void trainAllTestCases()
   {
      while (!reachedError && !reachedMaxIter) // keep going until we are under the error threshold or until max iterations
      {
         iterations++;
         double sumError = 0.0;
         for (int n = 0; n < nCases; n++)          // loop over test cases
         {
            setUpActivations(n);                   // set up training
            F0 = runOnce();                        // Run a forward pass of the network
            sumError = trainOnce(n, sumError);
         } // for (int n = 0; n < nCases; n++)
         avgError = sumError / (double) nCases;
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
/*
* If running without training: we can still run preloaded or random weights
*/
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
      for (int n = 0; n < nCases; n++)
      {
         setUpRunActivations(testCases[n]);
         runOnce(); // run the network
         outputs[n] = F0;
      } // for (int n = 0; n < nCases; n++)
   } // public static void runAllTestCases()

/**
* When in mode run, prints the output results (input table, truth table, output values)
* @param rr the result of running the entire network
*/
   public static void printResults()
   {
      System.out.println("--- Run Results ---");
/**
* Print each value in the input table (if the user wants to show the input table)
*/
      if (showInputTable)
      {
         System.out.println("Input Table:"); // this prints the inputs in order
         for (int n = 0; n < nCases; n++)
         {
            for (int k = 0; k < numInput; k++) System.out.print((int) testCases[n][k] + " ");
            System.out.println();
         }
      } // if (showInputTable)
/**
* Print each value in the truth table (if the user want to show the truth table)
*/
      if (showTruthTable && truth != null)
      {
         System.out.println("Truth Table:");
         for (int n = 0; n < nCases; n++)
         {
            System.out.println(String.format("%d %d -> %d", 
            (int) testCases[n][0], (int) testCases[n][1], (int) truth[n])); 
         }
      } // if (showTruthTable && rr.truths != null)
/**
* Prints the output values.
*/
      System.out.println("Actual Outputs:");
      for (int n = 0; n < nCases; n++)
      {
         System.out.println(String.format("Input: %d %d -> Output: %.6f", 
         (int) testCases[n][0], (int) testCases[n][1], outputs[n]));
      }
      System.out.println("--------------------");
   } // public static void printResults()

/**
* Puts it all together!
*/
   public static void main(String[] args)
   {
      setConfigParams();
      echoConfigParams();
      initializeRNG();
      allocateMemory();
      populateArrays();
      if (modeTrain)
      {
         trainAllTestCases();
         printTrainError();
      }
      runAllTestCases();
      printResults();
   } // public static void main(String[] args)
} // public class AB1Network