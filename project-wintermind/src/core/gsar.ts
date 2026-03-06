/**
 * WINTERMIND — GSAR: General Symbolic Arrays Reasoning
 *
 * Priority-rated symbolic grouping system.
 * Assigns priority scores to token n-grams and collapses high-confidence
 * recognized patterns into symbolic concept nodes, skipping redundant
 * per-token processing. Maintains a self-trained synonym/concept registry
 * that evolves across training steps.
 */

export interface ConceptNode {
  id: string;
  tokens: string[];          // canonical token sequence
  synonymGroups: string[][]; // dynamically accumulated equivalent patterns
  priority: number;          // 0..1, higher = skip detailed processing
  hitCount: number;
  conceptLabel: string;      // human-readable semantic label
}

export interface GSARResult {
  segments: GSARSegment[];
  skippedTokens: number;
  processedTokens: number;
  nodesActivated: string[];
  priorityMap: Map<number, number>; // tokenIdx → priority score
}

export interface GSARSegment {
  tokenIndices: number[];
  tokens: string[];
  conceptId: string | null;   // null = no concept match, process normally
  priority: number;
  symbolic: boolean;           // true = can skip dense processing
  label: string;
}

export class GSARModule {
  private concepts: Map<string, ConceptNode> = new Map();
  private ngramIndex: Map<string, string> = new Map(); // ngram key → conceptId
  private hitThreshold: number;
  private priorityThreshold: number;

  // Concept templates seeded from the "winter" example in the spec
  private static SEED_CONCEPTS: Array<{
    label: string;
    tokens: string[][];
  }> = [
    {
      label: 'winter_environment',
      tokens: [
        ['snow', 'cold', 'ice', 'frost', 'frozen', 'blizzard', 'winter', 'sleet', 'hail'],
        ['falling', 'snowflake', 'snowfall', 'flurry'],
        ['bitter', 'freezing', 'frigid', 'arctic'],
      ],
    },
    {
      label: 'learning_process',
      tokens: [
        ['learn', 'learns', 'learning', 'learned'],
        ['train', 'training', 'trained', 'fit', 'fitting'],
        ['optimize', 'optimizing', 'optimization'],
        ['converge', 'converging', 'convergence'],
      ],
    },
    {
      label: 'network_component',
      tokens: [
        ['layer', 'layers', 'neuron', 'neurons', 'weight', 'weights', 'bias'],
        ['gradient', 'gradients', 'backprop', 'backpropagation'],
        ['activation', 'relu', 'sigmoid', 'tanh', 'softmax'],
        ['attention', 'transformer', 'embedding', 'embedding'],
      ],
    },
    {
      label: 'sequential_data',
      tokens: [
        ['sequence', 'sequences', 'token', 'tokens', 'word', 'words'],
        ['context', 'position', 'step', 'time'],
        ['sentence', 'text', 'document', 'corpus'],
      ],
    },
    {
      label: 'loss_computation',
      tokens: [
        ['loss', 'error', 'cost', 'penalty'],
        ['cross', 'entropy', 'perplexity'],
        ['predict', 'prediction', 'predicted', 'inference'],
      ],
    },
  ];

  constructor(hitThreshold = 2, priorityThreshold = 0.5) {
    this.hitThreshold = hitThreshold;
    this.priorityThreshold = priorityThreshold;
    this.seedConcepts();
  }

  private seedConcepts() {
    for (const seed of GSARModule.SEED_CONCEPTS) {
      const id = seed.label;
      const allTokens = seed.tokens.flat();
      const node: ConceptNode = {
        id,
        tokens: allTokens,
        synonymGroups: seed.tokens,
        priority: 0.7,
        hitCount: 0,
        conceptLabel: seed.label.replace(/_/g, ' '),
      };
      this.concepts.set(id, node);
      // Index every token
      for (const tok of allTokens) {
        this.ngramIndex.set(tok.toLowerCase(), id);
      }
      // Index bigrams within groups
      for (const group of seed.tokens) {
        for (let i = 0; i < group.length - 1; i++) {
          const key = group[i] + '\u001f' + group[i + 1];
          this.ngramIndex.set(key, id);
        }
      }
    }
  }

  // ── Core reasoning pass ────────────────────────────────────────────────────

  reason(tokens: string[]): GSARResult {
    const segments: GSARSegment[] = [];
    const priorityMap = new Map<number, number>();
    const nodesActivated: string[] = [];
    let skippedTokens = 0;
    let processedTokens = 0;

    let i = 0;
    while (i < tokens.length) {
      const tok = tokens[i].toLowerCase();

      // Try bigram match first
      let matched = false;
      if (i < tokens.length - 1) {
        const bigramKey = tok + '\u001f' + tokens[i + 1].toLowerCase();
        const conceptId = this.ngramIndex.get(bigramKey);
        if (conceptId) {
          const node = this.concepts.get(conceptId)!;
          node.hitCount++;
          if (node.hitCount >= this.hitThreshold) node.priority = Math.min(node.priority + 0.02, 1.0);
          const symbolic = node.priority >= this.priorityThreshold;
          segments.push({
            tokenIndices: [i, i + 1],
            tokens: [tokens[i], tokens[i + 1]],
            conceptId,
            priority: node.priority,
            symbolic,
            label: node.conceptLabel,
          });
          priorityMap.set(i, node.priority);
          priorityMap.set(i + 1, node.priority);
          if (symbolic) skippedTokens += 2;
          else processedTokens += 2;
          if (!nodesActivated.includes(conceptId)) nodesActivated.push(conceptId);
          i += 2;
          matched = true;
        }
      }

      // Try unigram match
      if (!matched) {
        const conceptId = this.ngramIndex.get(tok);
        if (conceptId) {
          const node = this.concepts.get(conceptId)!;
          node.hitCount++;
          if (node.hitCount >= this.hitThreshold) node.priority = Math.min(node.priority + 0.01, 1.0);
          const symbolic = node.priority >= this.priorityThreshold;
          segments.push({
            tokenIndices: [i],
            tokens: [tokens[i]],
            conceptId,
            priority: node.priority,
            symbolic,
            label: node.conceptLabel,
          });
          priorityMap.set(i, node.priority);
          if (symbolic) skippedTokens++;
          else processedTokens++;
          if (!nodesActivated.includes(conceptId)) nodesActivated.push(conceptId);
        } else {
          // Unknown token — full processing required
          segments.push({
            tokenIndices: [i],
            tokens: [tokens[i]],
            conceptId: null,
            priority: 0,
            symbolic: false,
            label: 'unknown',
          });
          priorityMap.set(i, 0);
          processedTokens++;
        }
        i++;
      }
    }

    return { segments, skippedTokens, processedTokens, nodesActivated, priorityMap };
  }

  // ── Online learning: register new co-occurrence patterns ──────────────────

  learnCooccurrence(tokenA: string, tokenB: string, conceptId: string) {
    const key = tokenA.toLowerCase() + '\u001f' + tokenB.toLowerCase();
    if (!this.ngramIndex.has(key)) {
      this.ngramIndex.set(key, conceptId);
      const node = this.concepts.get(conceptId);
      if (node) {
        // Try to add to an existing synonym group or create a new one
        const aInGroup = node.synonymGroups.some((g) =>
          g.includes(tokenA.toLowerCase())
        );
        if (!aInGroup) {
          node.synonymGroups.push([tokenA.toLowerCase(), tokenB.toLowerCase()]);
          node.tokens.push(tokenA.toLowerCase(), tokenB.toLowerCase());
        }
      }
    }
  }

  // Promote a token sequence to a new concept dynamically
  registerConcept(label: string, tokenGroups: string[][]): string {
    const id = label.replace(/\s+/g, '_').toLowerCase() + '_' + Date.now();
    const allTokens = tokenGroups.flat();
    const node: ConceptNode = {
      id,
      tokens: allTokens,
      synonymGroups: tokenGroups,
      priority: 0.5,
      hitCount: 0,
      conceptLabel: label,
    };
    this.concepts.set(id, node);
    for (const tok of allTokens) this.ngramIndex.set(tok.toLowerCase(), id);
    return id;
  }

  getConcepts(): ConceptNode[] {
    return Array.from(this.concepts.values());
  }

  diagnostics(): Record<string, unknown> {
    const nodes = Array.from(this.concepts.values());
    return {
      totalConcepts: nodes.length,
      totalNgramKeys: this.ngramIndex.size,
      topConcepts: nodes
        .sort((a, b) => b.hitCount - a.hitCount)
        .slice(0, 5)
        .map((n) => ({ label: n.conceptLabel, hits: n.hitCount, priority: n.priority.toFixed(3) })),
    };
  }

  // Compute a "symbolic allocation ratio" — fraction of tokens saved by GSAR
  allocationRatio(result: GSARResult): number {
    const total = result.skippedTokens + result.processedTokens;
    return total === 0 ? 0 : result.skippedTokens / total;
  }
}
