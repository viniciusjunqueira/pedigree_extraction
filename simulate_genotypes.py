import numpy as np

def simulate_genotypes(genotyped_file='genotyped_1M.txt',
                      output_file='genotypes_1M.txt',
                      n_snps=50000,
                      maf_range=(0.05, 0.5)):
    """
    Simula genótipos para indivíduos genotipados

    Parameters:
    -----------
    genotyped_file : str
        Arquivo com IDs dos indivíduos genotipados
    output_file : str
        Arquivo de saída com genótipos
    n_snps : int
        Número de SNPs a simular
    maf_range : tuple
        Faixa de frequência do alelo menor (min, max)
    """

    print(f"Lendo arquivo de indivíduos genotipados: {genotyped_file}")

    # Ler IDs de forma mais eficiente
    ids = []
    with open(genotyped_file, 'r') as f:
        for line in f:
            ids.append(int(line.split()[0]))

    ids = np.array(ids, dtype=np.int32)
    n_individuals = len(ids)

    # Determinar largura FIXA do ID baseada no MAIOR ID possível
    max_id_width = len(str(ids.max()))

    print(f"Total de indivíduos: {n_individuals:,}")
    print(f"Largura fixa do ID: {max_id_width} caracteres")
    print(f"Simulando {n_snps:,} SNPs...")

    # Gerar MAFs aleatórias uma vez
    print("Gerando frequências alélicas...")
    mafs = np.random.uniform(maf_range[0], maf_range[1], size=n_snps)

    # Abrir arquivo para escrita
    print(f"Gerando genótipos e salvando em {output_file}...")
    with open(output_file, 'w') as f:
        # Processar em chunks para economizar memória
        chunk_size = 10000

        for chunk_start in range(0, n_individuals, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_individuals)
            chunk_ids = ids[chunk_start:chunk_end]
            n_chunk = len(chunk_ids)

            if chunk_start % 100000 == 0:
                print(f"  Progresso: {chunk_start:,}/{n_individuals:,} ({100*chunk_start/n_individuals:.1f}%)")

            # Gerar genótipos para este chunk
            genotypes = np.zeros((n_chunk, n_snps), dtype=np.int8)

            for i, maf in enumerate(mafs):
                # Probabilidades para genótipos assumindo Hardy-Weinberg
                p = maf
                q = 1 - p
                probs = [q**2, 2*p*q, p**2]
                genotypes[:, i] = np.random.choice([0, 1, 2], size=n_chunk, p=probs)

            # Escrever genótipos com LARGURA FIXA usando zero-padding
            for idx, animal_id in enumerate(chunk_ids):
                geno_string = ''.join(genotypes[idx].astype(str))
                # Zero-padding garante que NUNCA há espaço no início da linha
                # Todos os IDs terão exatamente max_id_width caracteres
                id_str = str(animal_id).zfill(max_id_width)
                f.write(f"{id_str} {geno_string}\n")

    print(f"\n✓ Arquivo de genótipos salvo: {output_file}")
    print(f"  Indivíduos: {n_individuals:,}")
    print(f"  SNPs por indivíduo: {n_snps:,}")
    file_size_mb = (n_individuals * (n_snps + max_id_width + 2)) / (1024**2)
    print(f"  Tamanho estimado do arquivo: ~{file_size_mb:.1f} MB")

    # Verificar alinhamento
    print("\nVerificando alinhamento das primeiras 10 linhas:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            parts = line.split(maxsplit=1)
            id_part = parts[0]
            snp_start = len(id_part) + 1
            print(f"  Linha {i+1}: ID='{id_part}' (len={len(id_part)}), SNPs começam na col {snp_start}")

    # Verificar últimas linhas também
    print("\nVerificando alinhamento das últimas 5 linhas:")
    with open(output_file, 'r') as f:
        all_lines = f.readlines()
        for i, line in enumerate(all_lines[-5:], start=len(all_lines)-4):
            parts = line.split(maxsplit=1)
            id_part = parts[0]
            snp_start = len(id_part) + 1
            print(f"  Linha {i}: ID='{id_part}' (len={len(id_part)}), SNPs começam na col {snp_start}")

    # Mostrar exemplo
    print("\nPrimeiras 3 linhas (primeiros 50 SNPs):")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            parts = line.strip().split(maxsplit=1)
            print(f"  {parts[0]} {parts[1][:50]}...")


if __name__ == "__main__":
    # Simular genótipos com 50k SNPs
    simulate_genotypes(
        genotyped_file='genotypes.txt_XrefID',
        output_file='genotypes.txt',
        n_snps=3,
        maf_range=(0.05, 0.5)
    )

    print("\n✓ Simulação de genótipos concluída!")
