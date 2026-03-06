/**
 * WINTERMIND — BYTE-PAIR ENCODING (BPE) TOKENIZER
 *
 * Full BPE implementation:
 *   - Corpus word-frequency counting
 *   - Character-level vocabulary initialization with </w> end-of-word marker
 *   - Iterative most-frequent-pair merging
 *   - Bidirectional vocabulary maps (token↔id)
 *   - Encode / Decode with correct </w> reconstruction
 *   - Special tokens: <pad>, <unk>, <bos>, <eos>
 */

export interface BPEVocab {
  tokenToId: Map<string, number>;
  idToToken: Map<number, string>;
  merges: Array<[string, string]>;
}

// ── Special tokens ──────────────────────────────────────────────────────────
const SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>'];
export const PAD_ID = 0;
export const UNK_ID = 1;
export const BOS_ID = 2;
export const EOS_ID = 3;

// ── Train BPE from raw text corpus ─────────────────────────────────────────

export function trainBPE(corpus: string, vocabSize: number): BPEVocab {
  const wordFreqs = getWordFrequencies(corpus);

  // Represent each word as space-separated chars + </w> end marker
  let vocab: Map<string, number> = new Map();
  for (const [word, freq] of wordFreqs) {
    if (word.length === 0) continue;
    const key = splitToChars(word);
    vocab.set(key, (vocab.get(key) ?? 0) + freq);
  }

  // Collect base characters
  const baseChars = new Set<string>();
  for (const key of vocab.keys()) {
    for (const sym of key.split(' ')) baseChars.add(sym);
  }

  const tokenToId = new Map<string, number>();
  const idToToken = new Map<number, string>();
  let nextId = 0;

  // Insert specials first
  for (const sp of SPECIAL) {
    tokenToId.set(sp, nextId);
    idToToken.set(nextId, sp);
    nextId++;
  }
  // Insert sorted base chars
  for (const c of Array.from(baseChars).sort()) {
    if (!tokenToId.has(c)) {
      tokenToId.set(c, nextId);
      idToToken.set(nextId, c);
      nextId++;
    }
  }

  const merges: Array<[string, string]> = [];
  const targetMerges = Math.max(0, vocabSize - nextId);

  for (let step = 0; step < targetMerges; step++) {
    const pairFreqs = getPairFrequencies(vocab);
    if (pairFreqs.size === 0) break;

    // Find the globally most frequent adjacent pair
    let best: [string, string] | null = null;
    let bestCount = -1;
    for (const [pairKey, count] of pairFreqs) {
      if (count > bestCount) {
        bestCount = count;
        const sep = pairKey.indexOf('\u0000');
        best = [pairKey.slice(0, sep), pairKey.slice(sep + 1)];
      }
    }
    if (!best || bestCount < 2) break;

    merges.push(best);
    const merged = best[0] + best[1];
    if (!tokenToId.has(merged)) {
      tokenToId.set(merged, nextId);
      idToToken.set(nextId, merged);
      nextId++;
    }

    vocab = applyMerge(vocab, best[0], best[1]);
  }

  return { tokenToId, idToToken, merges };
}

// ── Encode text → token ids ─────────────────────────────────────────────────

export function encode(text: string, vocab: BPEVocab): number[] {
  const words = text.trim().split(/\s+/).filter(w => w.length > 0);
  const ids: number[] = [BOS_ID];

  for (const word of words) {
    // Start as individual chars + </w>
    let symbols: string[] = word.split('').concat(['</w>']);
    // Merge the last char and </w> correctly: word → chars + </w> as last symbol
    symbols = [...word.split(''), '</w>'];

    // Apply merges in training order
    for (const [a, b] of vocab.merges) {
      let i = 0;
      while (i < symbols.length - 1) {
        if (symbols[i] === a && symbols[i + 1] === b) {
          symbols.splice(i, 2, a + b);
          // Don't increment — the new merged token may be mergeable with the next
        } else {
          i++;
        }
      }
    }

    for (const sym of symbols) {
      const id = vocab.tokenToId.get(sym) ?? UNK_ID;
      ids.push(id);
    }
  }

  ids.push(EOS_ID);
  return ids;
}

// ── Decode token ids → text ─────────────────────────────────────────────────
// </w> marks end-of-word: we emit a space after seeing it.
// Any token ending in </w> also triggers a space after reconstruction.

export function decode(ids: number[], vocab: BPEVocab): string {
  let result = '';
  for (const id of ids) {
    const tok = vocab.idToToken.get(id);
    if (!tok || SPECIAL.includes(tok)) continue;

    if (tok.endsWith('</w>')) {
      // Strip </w> and append space (word boundary)
      result += tok.slice(0, tok.length - 4) + ' ';
    } else if (tok === '</w>') {
      result += ' ';
    } else {
      result += tok;
    }
  }
  return result.trim();
}

// ── Word-level tokenization helper for display ──────────────────────────────

export function tokenizeForDisplay(text: string, vocab: BPEVocab): string[] {
  const ids = encode(text, vocab);
  return ids
    .filter(id => !SPECIAL.includes(vocab.idToToken.get(id) ?? ''))
    .map(id => vocab.idToToken.get(id) ?? `<${id}>`);
}

// ── Internal helpers ────────────────────────────────────────────────────────

function getWordFrequencies(text: string): Map<string, number> {
  const freq = new Map<string, number>();
  for (const word of text.toLowerCase().split(/\s+/)) {
    if (word.length === 0) continue;
    freq.set(word, (freq.get(word) ?? 0) + 1);
  }
  return freq;
}

// "hello" → "h e l l o </w>"  (space-separated for internal BPE representation)
function splitToChars(word: string): string {
  return word.split('').join(' ') + ' </w>';
}

function getPairFrequencies(vocab: Map<string, number>): Map<string, number> {
  const pairFreqs = new Map<string, number>();
  for (const [word, freq] of vocab) {
    const symbols = word.split(' ');
    for (let i = 0; i < symbols.length - 1; i++) {
      // Use null byte as separator (cannot appear in tokens)
      const key = symbols[i] + '\u0000' + symbols[i + 1];
      pairFreqs.set(key, (pairFreqs.get(key) ?? 0) + freq);
    }
  }
  return pairFreqs;
}

// Merge all occurrences of (a, b) → (a+b) in the vocabulary
function applyMerge(
  vocab: Map<string, number>,
  a: string,
  b: string
): Map<string, number> {
  const newVocab = new Map<string, number>();
  const target = a + ' ' + b; // what we're looking for in space-sep form
  const replacement = a + b;

  for (const [word, freq] of vocab) {
    // Replace ALL non-overlapping occurrences of "a b" with "ab"
    let newWord = '';
    const parts = word.split(' ');
    let i = 0;
    while (i < parts.length) {
      if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
        newWord += (newWord ? ' ' : '') + replacement;
        i += 2;
      } else {
        newWord += (newWord ? ' ' : '') + parts[i];
        i++;
      }
    }
    // Suppress TS unused variable warning
    void target;
    newVocab.set(newWord, (newVocab.get(newWord) ?? 0) + freq);
  }
  return newVocab;
}

// ── Bootstrap vocabulary (pre-trained on representative corpus) ─────────────

export function buildBootstrapVocab(): BPEVocab {
  const sampleCorpus = `
    the quick brown fox jumps over the lazy dog
    winter is cold and the snow falls softly on the ground
    the model learns patterns from data and generalizes well
    neural networks learn by adjusting weights through backpropagation
    language models predict the next token given prior context
    attention is all you need for transformer architectures
    the loss function measures how far predictions are from truth
    gradient descent minimizes loss by following the negative gradient
    tokenization breaks text into subword units for efficient processing
    embeddings map tokens into dense continuous vector representations
    recurrent networks maintain hidden state across sequence positions
    byte pair encoding creates efficient subword vocabularies from corpora
    zero redundancy optimizer shards memory across parallel processes
    nesterov momentum computes look ahead gradient for faster convergence
    hessian curvature provides second order information about the loss
    kernel fusion combines operations to reduce memory bandwidth overhead
    symbolic reasoning identifies concept patterns without redundant work
    self explanatory perception monitors prediction confidence in real time
    the winter wind blows cold through frozen snow covered trees at night
    ice and frost cover the ground in the bitter winter morning air
    deep learning requires careful optimization and strong regularization
    the gradient flows backward through the computational graph to update weights
    mixed precision training uses half and full precision for speed and stability
    distributed training splits the workload across many compute nodes efficiently
  `.trim();
  return trainBPE(sampleCorpus, 300);
}
