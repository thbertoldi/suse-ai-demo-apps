package main

import (
	"context"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.40.0"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthgrpc "google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	pb "github.com/suse/suse-ai-demo-apps/gateway/pb"
)

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func setupOTel(ctx context.Context, serviceName string) (func(), error) {
	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(semconv.SchemaURL, semconv.ServiceName(serviceName)),
	)
	if err != nil {
		return nil, err
	}

	traceExp, err := otlptracegrpc.New(ctx, otlptracegrpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(traceExp), sdktrace.WithResource(res))
	otel.SetTracerProvider(tp)

	metricExp, err := otlpmetricgrpc.New(ctx, otlpmetricgrpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExp)), sdkmetric.WithResource(res))
	otel.SetMeterProvider(mp)

	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return func() {
		_ = tp.Shutdown(ctx)
		_ = mp.Shutdown(ctx)
	}, nil
}

type gatewayServer struct {
	pb.UnimplementedDemoServiceServer
	ragConn   *grpc.ClientConn
	llmConn   *grpc.ClientConn
	agentConn *grpc.ClientConn
}

func (s *gatewayServer) Query(ctx context.Context, req *pb.QueryRequest) (*pb.QueryResponse, error) {
	client := pb.NewRAGServiceClient(s.ragConn)
	resp, err := client.Retrieve(ctx, &pb.RetrieveRequest{
		Query: req.Query,
		TopK:  req.TopK,
	})
	if err != nil {
		return nil, err
	}
	return &pb.QueryResponse{
		Answer:  resp.Answer,
		Sources: resp.Sources,
		Model:   resp.Model,
	}, nil
}

func (s *gatewayServer) Chat(ctx context.Context, req *pb.ChatRequest) (*pb.ChatResponse, error) {
	client := pb.NewLLMServiceClient(s.llmConn)
	resp, err := client.Generate(ctx, &pb.GenerateRequest{
		Prompt: req.Message,
	})
	if err != nil {
		return nil, err
	}
	return &pb.ChatResponse{
		Reply: resp.Text,
		Model: resp.Model,
	}, nil
}

func (s *gatewayServer) AgentChat(ctx context.Context, req *pb.AgentChatRequest) (*pb.AgentChatResponse, error) {
	client := pb.NewAgentServiceClient(s.agentConn)
	resp, err := client.Run(ctx, &pb.AgentRequest{
		Message: req.Message,
	})
	if err != nil {
		return nil, err
	}
	return &pb.AgentChatResponse{
		Reply:         resp.Reply,
		Model:         resp.Model,
		ToolCallsMade: resp.ToolCallsMade,
	}, nil
}

func main() {
	ctx := context.Background()

	serviceName := envOrDefault("OTEL_SERVICE_NAME", "gateway")
	shutdownOTel, err := setupOTel(ctx, serviceName)
	if err != nil {
		log.Fatalf("failed to setup OTel: %v", err)
	}
	defer shutdownOTel()

	listenAddr := envOrDefault("GRPC_LISTEN_ADDR", ":50051")
	ragAddr := envOrDefault("RAG_SERVICE_ADDR", "rag-service:50052")
	llmAddr := envOrDefault("LLM_SERVICE_ADDR", "llm-service:50053")

	ragConn, err := grpc.NewClient(ragAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithStatsHandler(otelgrpc.NewClientHandler()),
	)
	if err != nil {
		log.Fatalf("failed to connect to RAG service: %v", err)
	}
	defer ragConn.Close()

	llmConn, err := grpc.NewClient(llmAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithStatsHandler(otelgrpc.NewClientHandler()),
	)
	if err != nil {
		log.Fatalf("failed to connect to LLM service: %v", err)
	}
	defer llmConn.Close()

	agentAddr := envOrDefault("AGENT_SERVICE_ADDR", "agent-service:50054")

	agentConn, err := grpc.NewClient(agentAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithStatsHandler(otelgrpc.NewClientHandler()),
	)
	if err != nil {
		log.Fatalf("failed to connect to Agent service: %v", err)
	}
	defer agentConn.Close()

	srv := grpc.NewServer(grpc.StatsHandler(otelgrpc.NewServerHandler()))

	pb.RegisterDemoServiceServer(srv, &gatewayServer{ragConn: ragConn, llmConn: llmConn, agentConn: agentConn})

	healthSrv := healthgrpc.NewServer()
	healthpb.RegisterHealthServer(srv, healthSrv)
	healthSrv.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	healthSrv.SetServingStatus("demo.DemoService", healthpb.HealthCheckResponse_SERVING)
	healthSrv.SetServingStatus("demo.AgentService", healthpb.HealthCheckResponse_SERVING)

	lis, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	go func() {
		log.Printf("Gateway listening on %s", listenAddr)
		if err := srv.Serve(lis); err != nil {
			log.Fatalf("failed to serve: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	<-sigCh
	log.Println("Shutting down...")
	srv.GracefulStop()
}
