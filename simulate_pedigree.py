import numpy as np
import pandas as pd

def simulate_overlapping_generations_pedigree(n_individuals=10_000_000, 
                                               n_genotyped=1_000_000,
                                               n_founders=50_000,
                                               avg_progeny_per_parent=8,
                                               generation_overlap=5,
                                               output_ped='pedigree.txt',
                                               output_geno='genotyped.txt'):
    """
    Simula um pedigree com sobreposição de gerações para população animal
    
    Parameters:
    -----------
    n_individuals : int
        Número total de indivíduos no pedigree (10 milhões)
    n_genotyped : int
        Número de indivíduos genotipados (1 milhão, os mais recentes)
    n_founders : int
        Número de fundadores iniciais sem pais conhecidos
    avg_progeny_per_parent : int
        Média de progênies por parental
    generation_overlap : int
        Número de gerações que podem se sobrepor
    """
    
    print(f"Iniciando simulação de {n_individuals:,} indivíduos...")
    
    # Inicializar arrays para o pedigree
    ids = np.arange(1, n_individuals + 1, dtype=np.int32)
    sires = np.zeros(n_individuals, dtype=np.int32)
    dams = np.zeros(n_individuals, dtype=np.int32)
    
    # Fundadores (sem pais conhecidos)
    print(f"Criando {n_founders:,} fundadores...")
    sires[:n_founders] = 0
    dams[:n_founders] = 0
    
    # Pool de parentais disponíveis por geração
    # Começamos com os fundadores
    current_generation = 0
    generation_pools = {0: np.arange(1, n_founders + 1)}
    
    # Gerar descendentes
    print("Gerando descendentes com sobreposição de gerações...")
    current_id = n_founders + 1
    
    # Usar chunks para processar em blocos
    chunk_size = 100_000
    total_generated = 0
    total_to_generate = n_individuals - n_founders
    
    while current_id <= n_individuals:
        # Imprimir progresso a cada 1M
        if total_generated % 1_000_000 == 0 and total_generated > 0:
            print(f"  Progresso: {total_generated:,}/{total_to_generate:,} ({100*total_generated/total_to_generate:.1f}%)")
        
        # Determinar quais gerações estão ativas para reprodução
        active_generations = range(max(0, current_generation - generation_overlap + 1), 
                                  current_generation + 1)
        
        # Combinar pools de todas as gerações ativas
        available_parents = np.concatenate([generation_pools[gen] 
                                           for gen in active_generations 
                                           if gen in generation_pools])
        
        if len(available_parents) == 0:
            break
        
        # Número de descendentes nesta rodada
        n_offspring = min(chunk_size, n_individuals - current_id + 1)
        
        # Selecionar pais aleatoriamente
        # Assumindo que machos e fêmeas estão misturados no pool
        parent_indices = np.random.choice(len(available_parents), 
                                        size=(n_offspring, 2), 
                                        replace=True)
        
        selected_sires = available_parents[parent_indices[:, 0]]
        selected_dams = available_parents[parent_indices[:, 1]]
        
        # Garantir que sire != dam
        same_parent = selected_sires == selected_dams
        while np.any(same_parent):
            new_dams = np.random.choice(available_parents, size=np.sum(same_parent))
            selected_dams[same_parent] = new_dams
            same_parent = selected_sires == selected_dams
        
        # Atribuir ao pedigree
        end_id = current_id + n_offspring
        sires[current_id-1:end_id-1] = selected_sires
        dams[current_id-1:end_id-1] = selected_dams
        
        # Adicionar nova geração ao pool
        current_generation += 1
        generation_pools[current_generation] = np.arange(current_id, end_id)
        
        # Remover gerações muito antigas do pool (para economizar memória)
        old_gen = current_generation - generation_overlap - 1
        if old_gen in generation_pools:
            del generation_pools[old_gen]
        
        current_id = end_id
        total_generated += n_offspring
    
    print(f"Pedigree completo com {n_individuals:,} indivíduos gerados.")
    
    # Salvar arquivo de pedigree
    print(f"Salvando arquivo de pedigree em {output_ped}...")
    pedigree_df = pd.DataFrame({
        'id': ids,
        'sire': sires,
        'dam': dams
    })
    
    pedigree_df.to_csv(output_ped, sep=' ', index=False, header=False)
    print(f"Arquivo de pedigree salvo: {output_ped}")
    
    # Selecionar os indivíduos mais recentes para genotipagem
    print(f"Selecionando {n_genotyped:,} indivíduos mais recentes para genotipagem...")
    genotyped_ids = ids[-n_genotyped:]
    
    # Salvar arquivo de genótipos (id repetido em duas colunas)
    print(f"Salvando arquivo de genotipados em {output_geno}...")
    genotyped_df = pd.DataFrame({
        'id1': genotyped_ids,
        'id2': genotyped_ids
    })
    
    genotyped_df.to_csv(output_geno, sep=' ', index=False, header=False)
    print(f"Arquivo de genotipados salvo: {output_geno}")
    
    # Estatísticas
    print("\n=== Estatísticas do Pedigree ===")
    print(f"Total de indivíduos: {n_individuals:,}")
    print(f"Fundadores (sem pais): {n_founders:,}")
    print(f"Descendentes: {n_individuals - n_founders:,}")
    print(f"Indivíduos genotipados: {n_genotyped:,}")
    print(f"Proporção genotipada: {n_genotyped/n_individuals*100:.2f}%")
    
    # Verificar completude do pedigree
    non_zero_parents = np.sum((sires > 0) & (dams > 0))
    print(f"Indivíduos com ambos os pais conhecidos: {non_zero_parents:,}")
    print(f"Taxa de completude: {non_zero_parents/(n_individuals-n_founders)*100:.2f}%")
    
    return pedigree_df, genotyped_df


if __name__ == "__main__":
    # Executar a simulação
    pedigree, genotyped = simulate_overlapping_generations_pedigree(
        n_individuals=10_000_000,
        n_genotyped=1_000_000,
        n_founders=50_000,
        avg_progeny_per_parent=4,
        generation_overlap=5,
        output_ped='pedigree.txt',
        output_geno='genotypes.txt_XrefID'
    )
    
    print("\n✓ Simulação concluída com sucesso!")
    print("\nPrimeiras 10 linhas do pedigree:")
    print(pedigree.head(10))
    print("\nÚltimas 10 linhas do pedigree:")
    print(pedigree.tail(10))
    print("\nPrimeiras 10 linhas dos genotipados:")
    print(genotyped.head(10))
