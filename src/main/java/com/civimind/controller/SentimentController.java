package com.civimind.controller;

import com.civimind.dto.SentimentRequest;
import com.civimind.dto.SentimentResponse;
import com.civimind.service.SentimentService;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

@RestController
@RequestMapping("/api/sentiment")
public class SentimentController {

    private final SentimentService sentimentService;

    public SentimentController(SentimentService sentimentService) {
        this.sentimentService = sentimentService;
    }

    @GetMapping
    public SentimentResponse analyzeGet(
            @RequestParam String text
    ) {
        return analyze(text);
    }

    @PostMapping
    public SentimentResponse analyzePost(
            @RequestBody SentimentRequest request
    ) {
        return analyze(request.text());
    }

    private SentimentResponse analyze(String text) {

        if (text == null || text.isBlank()) {
            throw new ResponseStatusException(
                    HttpStatus.BAD_REQUEST,
                    "Текст обращения не должен быть пустым"
            );
        }

        if (text.length() > 10000) {
            throw new ResponseStatusException(
                    HttpStatus.BAD_REQUEST,
                    "Максимальная длина текста — 10000 символов"
            );
        }

        return sentimentService.analyze(text);
    }
}