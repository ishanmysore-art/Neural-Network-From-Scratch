import java.io.*;
import java.nio.file.*;

/**
 * ProcessPelArray.java
 *
 * Pipeline:
 * 1. Load 3024x4032 grayscale byte image
 * 2. Coarse crop to remove background
 * 3. Ones-complement if background is white
 * 4. Compute center of mass (COM)
 * 5. Fixed-size crop around COM
 * 6. Write processed grayscale bytes to output file
 */
public class ProcessPelArraySafe {

    // ORIGINAL IMAGE SIZE
    private static final int ORIG_WIDTH  = 3024;
    private static final int ORIG_HEIGHT = 4032;

    // FINAL CROPPED SIZE (TUNE THIS)
    private static final int FINAL_WIDTH  = 3500;
    private static final int FINAL_HEIGHT = 3500;
    private static final int SCALED_WIDTH = 256;
    private static final int SCALED_HEIGHT = 320;

    public static void main(String[] args) throws IOException {

        if (args.length != 2) {
            System.out.println("Usage: java ProcessPelArray <input.bin> <output.bin>");
            return;
        }

        String inputFile  = args[0];
        String outputFile = args[1];

        byte[] rawBytes = Files.readAllBytes(Paths.get(inputFile));

        if (rawBytes.length != ORIG_WIDTH * ORIG_HEIGHT) {
            System.err.println("ERROR: File size does not match expected dimensions.");
            return;
        }

        /* ---------------------------------------------------------
         * 1. Load byte data into PelArray (grayscale → RGB pels)
         * --------------------------------------------------------- */
        int[][] pels = new int[ORIG_HEIGHT][ORIG_WIDTH];

        for (int y = 0; y < ORIG_HEIGHT; y++) {
            for (int x = 0; x < ORIG_WIDTH; x++) {
                int v = rawBytes[y * ORIG_WIDTH + x] & 0xFF;
                pels[y][x] = (v << 16) | (v << 8) | v;
            }
        }

        PelArray img = new PelArray(pels);
    
        /* ---------------------------------------------------------
         * 3. ONES COMPLEMENT if background is white
         *    Sample corner pixel
         * ------------------------------------------------------- */

        for (int y = 0; y < img.getHeight(); y++) {
            for (int x = 0; x < img.getWidth(); x++) {
                int gray = pels[y][x] & 0xFF;
                if (gray < 125) gray = 0;        // force black
                if (gray > 125) gray = 255;     // optional force white
                pels[y][x] = (gray << 16) | (gray << 8) | gray;
            }
        }
        img = new PelArray(pels);
        img = img.onesComplimentImage();
        
        /* ---------------------------------------------------------
         * 4. CENTER OF MASS (COM)
         * --------------------------------------------------------- */
        int comX = img.getXcom();
        int comY = img.getYcom();

        /* ---------------------------------------------------------
         * 5. FIXED-SIZE CROP around COM
         * --------------------------------------------------------- */
        int halfW = FINAL_WIDTH  / 2;
        int halfH = FINAL_HEIGHT / 2;

        int x0 = Math.max(0, comX - halfW);
        int y0 = Math.max(0, comY - halfH);
        int x1 = Math.min(img.getWidth()  - 1, x0 + FINAL_WIDTH  - 1);
        int y1 = Math.min(img.getHeight() - 1, y0 + FINAL_HEIGHT - 1);

        img = img.crop(x0, y0, x1, y1);

        img = img.scale(SCALED_WIDTH, SCALED_HEIGHT);

        /* ---------------------------------------------------------
         * 6. WRITE OUTPUT as 8-bit grayscale bytes
         * --------------------------------------------------------- */
        img = img.grayScaleImage();
        int[][] out = img.getPelArray();

        byte[] outBytes = new byte[SCALED_WIDTH * SCALED_HEIGHT];

        for (int y = 0; y < SCALED_HEIGHT; y++) {
            for (int x = 0; x < SCALED_WIDTH; x++) {
                outBytes[y * SCALED_WIDTH + x] = (byte)(out[y][x] & 0xFF);
            }
        }

    
        Files.write(Paths.get(outputFile), outBytes);
    }
}