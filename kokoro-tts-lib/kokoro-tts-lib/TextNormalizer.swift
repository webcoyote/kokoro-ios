//
//  kokoro-tts-lib
//

class TextNormalizer {
    private init() {}
    
    private static func reSub(_ text: String, _ pattern: String, _ replacement: String, caseInsensitive: Bool = false) -> String {
        let options: NSRegularExpression.Options = caseInsensitive ? [.caseInsensitive] : []
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { return text }
        
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: replacement)
    }
    
    private static func reSubYeahPattern(_ input: String) -> String {
       guard let regex = try? NSRegularExpression(pattern: #"(?i)\b(y)eah?\b"#, options: []) else {
           return input
       }
       let inputRange = NSRange(input.startIndex..<input.endIndex, in: input)
       
       var result = input
       let matches = regex.matches(in: input, options: [], range: inputRange)

        for match in matches.reversed() {
           guard match.numberOfRanges >= 2 else { continue }
           let wholeRange = match.range(at: 0)
           let yRange = match.range(at: 1) // capturing group #1 => the 'y'
           
           if let whole = Range(wholeRange, in: result),
              let ySub = Range(yRange, in: result) {
               let yValue = String(result[ySub]) // e.g. "y" or "Y"
               let replacement = yValue + "e'a"
               result.replaceSubrange(whole, with: replacement)
           }
       }
       return result
    }
    
    static func normalizeText(text: String) -> String {
        var normalizedText = ""
        normalizedText = text
            .replacingOccurrences(of: String(UnicodeScalar(8216)!), with: "'")
            .replacingOccurrences(of: String(UnicodeScalar(8217)!), with: "'")
            .replacingOccurrences(of: "«", with: String(UnicodeScalar(8220)!))
            .replacingOccurrences(of: "»", with: String(UnicodeScalar(8221)!))
            .replacingOccurrences(of: String(UnicodeScalar(8220)!), with: "\"")
            .replacingOccurrences(of: String(UnicodeScalar(8221)!), with: "\"")
            .replacingOccurrences(of: "(", with: "«")
            .replacingOccurrences(of: ")", with: "»")
        
        for (source, target) in zip("、。！，：；？", ",.!,:;?") {
            let sourceStr = String(source)
            let targetStr = String(target) + " "
            normalizedText = normalizedText.replacingOccurrences(of: sourceStr, with: targetStr)
        }
        
        normalizedText = reSub(normalizedText, #"[^\S\n]"#, " ")
        normalizedText = reSub(normalizedText, #" {2,}"#, " ")
        normalizedText = reSub(normalizedText, #"(?<=\n) +(?=\n)"#, "")
        normalizedText = reSub(normalizedText, #"\bD[Rr]\.(?= [A-Z])"#, "Doctor")
        normalizedText = reSub(normalizedText, #"\b(?:Mr\.|MR\.(?= [A-Z]))"#, "Mister")
        normalizedText = reSub(normalizedText, #"\b(?:Ms\.|MS\.(?= [A-Z]))"#, "Miss")
        normalizedText = reSub(normalizedText, #"\b(?:Mrs\.|MRS\.(?= [A-Z]))"#, "Mrs")
        normalizedText = reSub(normalizedText, #"\betc\.(?! [A-Z])"#, "etc")
        normalizedText = reSubYeahPattern(normalizedText)

        return normalizedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
