// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "demucs-mlx-swift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "DemucsMLX", targets: ["DemucsMLX"]),
        .executable(name: "demucs-mlx-swift", targets: ["DemucsCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.31.3")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "DemucsMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "DemucsCLILib",
            dependencies: [
                "DemucsMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "DemucsCLI",
            dependencies: ["DemucsCLILib"]
        ),
        .testTarget(
            name: "DemucsMLXTests",
            dependencies: ["DemucsMLX"]
        ),
    ]
)
