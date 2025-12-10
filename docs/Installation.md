## ⚙️ Installation
>To more complete installation instructions and usage, please refer below.

1. **Clone the repository**
    ```bash
    git clone https://github.com/GENTEL-lab/AuroBind.git
    cd src
    ```

2. **Create and activate the environment(recommended)**
    ```bash
    conda env create -f environment.yaml
    conda activate aurobind
    ```

3. **Clone NVIDIA Cutlass repository**
   
    ```bash
    git clone https://github.com/NVIDIA/cutlass --depth 1
    # Set environment variable for Cutlass path, or set it in predict.sh
    conda env config vars set CUTLASS_PATH=$PWD/cutlass
   ```
    
    **If the kernel compilation fails, please check the GCC version to ensure it is 11.2 or higher.**
    
    ~~~   bash
    # Optional: Install gcc and g++ 11.2 with conda if your g++ version is lower than 11.2
    conda install gxx_linux-64=11.2 gxx_impl_linux-64=11.2 gcc_linux-64=11.2 gcc_impl_linux-64=11.2 libstdcxx-ng=11.2
    conda install gcc=11.2 gxx=11.2 -c conda-forge
    ~~~
    
    
    
4. **(Optional) Download  Cache Data Manually**<br>To manually download from [Google Drive](https://drive.google.com/drive/folders/1DNZDXewqFibbP0erslQlm6Sp0qVQG1Ad?usp=sharing).Place the downloaded files in the `src/cache_data/` directory before running inference.
    
    Your directory should look like:
    ```
    cache_data/
    ├── model.pt
    ├── ccd.pkl
    ├── unique_protein_sequences.fasta
    ├── unique_nucleic_acid_sequences.fasta
    ├── protein_id_groups.json
    └── nucleic_acid_id_groups.json
    ```
    