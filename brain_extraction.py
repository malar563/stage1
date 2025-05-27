import nibabel as nib
from totalsegmentator.python_api import totalsegmentator


def segment_brain(input_path="nifti/6_cow_angio__06__hv36__3.nii.gz", output_path="segm", fast=False):
    if __name__ == "__main__":

        input_img = nib.load(input_path)
        output_img = totalsegmentator(input_img, fast=fast, roi_subset=["brain"]) # Mettre False pour plus vite
        nib.save(output_img, output_path)
        # Brain is labeled with the number 90
        # Skull is labeled with the number 91
        

segment_brain()