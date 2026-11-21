FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

COPY target/civimind-1.0.0.jar app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]


# Этап 1. Создаём компактную Java Runtime
FROM eclipse-temurin:17-jdk-alpine AS builder

RUN jlink \
    --add-modules java.base,java.logging,java.management,java.naming,java.instrument,java.sql,java.xml,java.desktop,java.security.jgss,jdk.unsupported,jdk.crypto.ec \
    --strip-debug \
    --no-man-pages \
    --no-header-files \
    --compress=2 \
    --output /custom-jre

# Этап 2. Собираем минимальный образ приложения
FROM alpine:3.20

RUN apk add --no-cache libstdc++

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY --from=builder /custom-jre /opt/java

WORKDIR /app

COPY target/civimind-1.0.0.jar app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
