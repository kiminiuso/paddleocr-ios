//
//  Utils.h
//  CardRecognize
//
//  Created by kiminiuso on R 3/06/17.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface Utils : NSObject

+ (NSArray *)readLabelsFromFile:(NSString *)labelFilePath;

@end

NS_ASSUME_NONNULL_END
