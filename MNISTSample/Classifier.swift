import UIKit
import CoreML

class Classifier {
    private let model: Model

    init() {
        let assetPath = Bundle.main.url(forResource: "Model", withExtension:"mlmodelc")
        self.model = try! Model(contentsOf: assetPath!)
    }

    // TODO: Improve CoreML types using `coremltools`.
    // https://qiita.com/shu223/items/da21a3f6f8f55ff01041
    func run(_ image: UIImage) throws -> Int {
        let out: ModelOutput = try self.model.prediction(input_1: image.grayscaleToMLArray())
        let arr = (0...9).map { ($0, Double(truncating: out._17[$0])) }
        let guess = arr.max(by: { $0.1 < $01.1 })!.0
        print(arr)
        print("guess =", guess)
        return guess
    }
}

extension UIImage {
    fileprivate func grayscaleToMLArray() -> MLMultiArray {
        let mlArray = try! MLMultiArray(
            shape: [1 /* color */, self.size.width as NSNumber, self.size.height as NSNumber],
            dataType: MLMultiArrayDataType.double
        )
        let imagePixels = (self.cgImage?.dataProvider?.data as Data?)
            .map { Array($0).map(Double.init) }
            ?? []
        mlArray.dataPointer.initializeMemory(as: Double.self, from: imagePixels, count: imagePixels.count)
        return mlArray
    }
}
