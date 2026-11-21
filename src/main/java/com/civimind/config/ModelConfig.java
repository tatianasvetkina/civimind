package com.civimind.config;

import com.civimind.model.SentimentModel;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.io.InputStream;

@Configuration
public class ModelConfig {

    private static final Logger log =
            LoggerFactory.getLogger(ModelConfig.class);

    @Bean
    public SentimentModel sentimentModel(ObjectMapper objectMapper)
            throws IOException {

        ClassPathResource resource =
                new ClassPathResource("ml/sentiment-model.json");

        try (InputStream inputStream = resource.getInputStream()) {

            SentimentModel model = objectMapper.readValue(
                    inputStream,
                    SentimentModel.class
            );

            log.info("CiviMind AI model loaded successfully");
            log.info("Model version: {}", model.version());
            log.info("Algorithm: {}", model.algorithm());
            log.info("Training samples: {}", model.trainingSamples());
            log.info("Features: {}", model.vocabulary().size());

            return model;
        }
    }
}