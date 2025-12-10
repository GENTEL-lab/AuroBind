## YAML format(Recommended)

The YAML format is more flexible and allows for more complex inputs, particularly around covalent bonds. The schema of the YAML is the following:

```yaml
sequences:
    - ENTITY_TYPE:
        id: CHAIN_ID 
        sequence: SEQUENCE    # only for protein, dna, rna
        smiles: 'SMILES'        # only for ligand, exclusive with ccd
        ccd: CCD              # only for ligand, exclusive with smiles
        msa: MSA_PATH         # only for protein
        modifications:
          - position: RES_IDX   # index of residue, starting from 1
            ccd: CCD            # CCD code of the modified residue

    - ENTITY_TYPE:
        id: [CHAIN_ID, CHAIN_ID]    # multiple ids in case of multiple identical entities
        ...

```
------

Each entry in `sequences` corresponds to a unique chain or molecule in the input. The `ENTITY_TYPE` is assigned based on the type of molecule: `protein` for proteins, and `ligand` for small molecules. For proteins, a `sequence` attribute is required, whereas for ligands, either a `smiles` or `ccd` attribute is used. The `CHAIN_ID` serves as the unique identifier for each chain or molecule and should be represented as a list if there are multiple identical entities in the structure.

For proteins, the `msa` key is required by default to specify the multiple sequence alignment (MSA). However, you can skip providing the MSA by using the `--use_msa_server` flag, which will auto-generate the MSA using the MMseqs2 server. If you have a precomputed MSA, specify it using the `msa` attribute and provide the path to the `.a3m` file. In cases where you explicitly want to run the model in single-sequence mode (though this is generally not recommended as it may degrade model performance), use the special keyword `empty` for the protein's `msa` (e.g., `msa: empty`). For custom MSAs, you may also use a CSV format with two columns: `sequence` for the protein sequences and `key`, a unique identifier for matching rows across CSV files of each protein chain.

The `modifications` field is optional and allows you to specify modified residues within polymers (`protein`). This includes the `position` field, which indicates the index (starting from 1) of the residue, and `ccd`, which is the CCD code for the modified residue. Currently, this field is only supported for CCD ligands.

> **Note**: The inference data pipeline is adapted from [Boltz-1](https://github.com/jwohlwend/boltz) and  [IntelliFold](https://github.com/IntelliGen-AI/IntelliFold.git).


As an example:

```yaml
sequences:
- protein:
    id: A
    modifications:
    - ccd: SEP
      position: 62
    msa: ./examples/msas/7wcf_A.a3m
    sequence: NLYFQSNAMKHCPITYEKISDQENYSQRGLHLLSPQLKNLSPLDLSADEQRQEAIARVGKMSVQGVQKKLSAKLKIKEGCFEIVDQYGQYILKPQSDIYPELPENEAITMTLAKTIGLEVPVHGLVYSKDNSLTYFIKRFDRIGHNKKLALEDFAQLSGEDRHTKYKSSMEKVIAVIEQFCTFPKIEFVKLFKLTLFNFLVGNEEMHLKNFSLITKDRKISISPAYDLLNSTIAQKNTKEELALPLKGKKNNLTKSDFLKYFAIEKLGLNQNVIDGIVQEFHQVIPKWQELIGFSFLSQEMQEKYLELLEQRCKRLNFFD
- ligand:
    ccd: ACP
    id: B
version: 1
```


