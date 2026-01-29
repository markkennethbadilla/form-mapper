Here is the **"Sacrificial Form Strategy"**. This is the absolute easiest, most headache-free way to do this. No math, no measuring tape, no guessing.

You are going to waste **one** real form to get perfect alignment forever.

### Phase 1: Create the "Map" (Do this on your laptop)

You need a grid PDF. Don't draw it manually.

1. **Copy** the code below.
2. **Paste** it into a text file and save it as `grid.html`.
3. **Open** it in your browser.
4. **Print to PDF**.
* **Important:** In print settings, set Scale to **"Custom: 100"** or **"Default"**. Do not use "Fit to Printable Area".



```html
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; padding: 0; font-family: sans-serif; font-size: 9px; }
        /* Setup for Standard US Letter (8.5 x 11) */
        .page { width: 8.5in; height: 11in; position: relative; overflow: hidden; }
        .v-line { position: absolute; top: 0; bottom: 0; border-left: 0.5px solid red; }
        .h-line { position: absolute; left: 0; right: 0; border-top: 0.5px solid red; }
        .coord { position: absolute; color: red; font-weight: bold; padding: 1px; }
    </style>
</head>
<body>
    <div class="page">
        <script>
            // This draws a line every 0.25 inches (Quarter Inch)
            const step = 0.25; 
            for (let x = 0; x < 8.5; x += step) {
                document.write(`<div class="v-line" style="left: ${x}in;"></div>`);
                for (let y = 0; y < 11; y += step) {
                    if (x === 0) document.write(`<div class="h-line" style="top: ${y}in;"></div>`);
                    // Only label every 0.5 inches to keep it clean
                    if (x % 0.5 === 0 && y % 0.5 === 0) {
                        document.write(`<div class="coord" style="left:${x+0.02}in; top:${y+0.02}in;">${x}, ${y}</div>`);
                    }
                }
            }
        </script>
    </div>
</body>
</html>

```

---

### Phase 2: The "Sacrifice" (Send this to your Coworker)

Send that PDF to your coworker with these **exact instructions**:

1. "Take **ONE** real, physical blank form."
2. "Put it in the printer tray."
3. "Print this Grid PDF **directly onto the form**." (Ensure they print at 100% Scale).
4. "Send me a high-quality photo of the form."

---

### Phase 3: The "Vibe Coding" (Your Job)

Open the photo they sent you. You will see red grid lines printed on top of the text boxes.

**Example:**
You look at the **"First Name"** box. You see the top-left corner of that box aligns with the numbers **1.5** (horizontal) and **3.0** (vertical).

Now, update your Next.js code. Since you are using a PDF library (like `pdf-lib` or `react-pdf`), you need to convert **Inches** to **Points**.

* **1 inch = 72 points**

**Your Code:**

```javascript
// Function to convert inches to points (so you don't have to do math)
const inch = (num) => num * 72;

// The PDF generation part
const page = pdfDoc.addPage([inch(8.5), inch(11)]); // Standard Letter Size

// Field: First Name
// You saw "1.5, 3.0" on the photo. You type "1.5, 3.0" here.
page.drawText('Juan Dela Cruz', {
    x: inch(1.5), 
    y: inch(11 - 3.0), // Note: PDF coordinates often start from BOTTOM. 
                       // If your text appears upside down or at the bottom, 
                       // do (PageHeight - Y_Value).
    size: 12
});

```

### Phase 4: The Final Print

1. Generate the PDF with your filled data.
2. Send it to your coworker.
3. They print it on a **fresh** blank form.

**Why this works:**
Because you printed the grid *using their specific printer*, any misalignment of that printer is already baked into the grid. If their printer pulls the paper 2mm to the left, the grid moved 2mm to the left, and your data will move 2mm to the left. **It cancels out perfectly.**