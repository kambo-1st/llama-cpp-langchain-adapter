<?php

namespace Kambo\LLamaCPPLangchainAdapter;

use Kambo\LLamaCPP\LLamaCPP;
use Kambo\LLamaCPP\Context;
use Kambo\LLamaCPP\Parameters\ModelParameters;
use Kambo\LLamaCPP\Parameters\GenerationParameters;

class LLamaCPPLangchainAdapter
{

    private LLamaCPP $llamaCPP;

    private int $numberOfThreads;

    public function __construct(array $config = [])
    {
        $context = Context::createWithParameter(
            new ModelParameters(
                modelPath:$config['model_path'],
                nCtx:$config['n_ctx'],
                nParts:$config['n_parts'],
                seed:$config['seed'],
                f16KV:$config['f16_kv'],
                logitsAll:$config['logits_all'],
                vocabOnly:$config['vocab_only'],
                useMlock:$config['use_mlock'],
            )
        );

        $this->numberOfThreads = $config['n_threads'] ?? 8;
        $this->llamaCPP = new LLamaCPP($context);
    }

    public function predict(string $prompt, array $parameters = []): string
    {
        return $this->llamaCPP->generateAll(
            $prompt,
            new GenerationParameters(
                predictLength: $parameters['max_tokens'],
                topP: $parameters['top_p'],
                topK: $parameters['top_k'],
                temperature: $parameters['temperature'],
                repeatPenalty: $parameters['repeat_penalty'],
                noOfThreads:$this->numberOfThreads,
            )
        );
    }

    public static function create(array $config = []): self
    {
        return new self($config);
    }
}
