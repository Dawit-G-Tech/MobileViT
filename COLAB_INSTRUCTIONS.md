# Google Colab Quick Start Guide

## Step 1: Open the Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Upload notebook"
3. Upload `MobileViT_CIFAR10.ipynb`

## Step 2: Enable GPU (Recommended)
1. Click "Runtime" → "Change runtime type"
2. Set "Hardware accelerator" to "GPU" (T4 is fine)
3. Click "Save"

## Step 3: Run the Setup Cells
1. Run the first cell to install dependencies
2. Run the second cell to import libraries and check GPU

## Step 4: Upload MobileViT Code
1. When you reach the "Upload MobileViT Implementation" cell, click "Choose Files"
2. Select and upload `mobilevit.py` from your local machine
3. Wait for the upload to complete
4. Run the cell - you should see "MobileViT module loaded successfully!"

## Step 5: Run Training
1. Run all remaining cells sequentially
2. The notebook will:
   - Download CIFAR-10 dataset automatically
   - Create the MobileViT model
   - Train for 100 epochs (this may take 1-2 hours on GPU)
   - Display training progress with progress bars
   - Save the best model checkpoint
   - Visualize training curves
   - Show sample predictions

## Tips
- **Monitor Training**: Watch the progress bars and epoch summaries
- **Early Stopping**: You can interrupt training early (Ctrl+C) if needed
- **Model Variants**: Change `model_variant = 'xxs'` to `'xs'` or `'s'` for larger models
- **Save Outputs**: Download the `best_model.pth` file to save your trained model
- **Resume Training**: You can modify the notebook to load a checkpoint and continue training

## Expected Runtime
- **MobileViT-XXS**: ~1-1.5 hours on T4 GPU for 100 epochs
- **MobileViT-XS**: ~1.5-2 hours on T4 GPU for 100 epochs  
- **MobileViT-S**: ~2-2.5 hours on T4 GPU for 100 epochs

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mobilevit'"
**Solution**: Make sure you uploaded `mobilevit.py` in the upload cell and it completed successfully.

### Issue: Out of Memory Error
**Solution**: 
- Reduce batch size (change `batch_size = 128` to `batch_size = 64`)
- Use a smaller model variant (`xxs` instead of `s`)

### Issue: Training is too slow
**Solution**:
- Make sure GPU is enabled (check Runtime → Change runtime type)
- Reduce number of epochs for testing
- Use a smaller model variant

### Issue: Colab disconnects during training
**Solution**:
- Keep the browser tab active
- Consider saving checkpoints more frequently (modify the training loop)
- Use Colab Pro for longer sessions

## Alternative: Copy-Paste Method

If file upload doesn't work, you can copy-paste the `mobilevit.py` content directly into a code cell:

1. Open `mobilevit.py` in a text editor
2. Copy all the content
3. In Colab, create a new code cell
4. Paste the content
5. Run the cell
6. Then continue with the rest of the notebook
